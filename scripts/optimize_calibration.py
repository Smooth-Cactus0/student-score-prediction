import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

def optimize_calibration():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    
    oof_path = os.path.join(sub_dir, 'oof_v21_mega_stacking.csv')
    df = pd.read_csv(oof_path)
    
    y_true = df['exam_score']
    y_pred = df['pred']
    
    current_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Current V21 RMSE: {current_rmse:.4f}")
    
    # 1. Simple Linear Calibration: a * pred + b
    def linear_obj(params):
        a, b = params
        calibrated = a * y_pred + b
        return np.sqrt(mean_squared_error(y_true, calibrated))
    
    res = minimize(linear_obj, [1.0, 0.0], method='Nelder-Mead')
    print(f"\nLinear Calibration (a * pred + b):")
    print(f"Scale (a): {res.x[0]:.4f}, Shift (b): {res.x[1]:.4f}")
    print(f"New RMSE: {res.fun:.4f}")
    
    # 2. Stretch Calibration around Mean: (pred - mean) * scale + mean
    mean_val = y_pred.mean()
    def stretch_obj(scale):
        calibrated = (y_pred - mean_val) * scale + mean_val
        return np.sqrt(mean_squared_error(y_true, calibrated))
    
    res_stretch = minimize(stretch_obj, [1.0], method='Nelder-Mead')
    print(f"\nStretch Calibration ((pred - mean) * scale + mean):")
    print(f"Scale: {res_stretch.x[0]:.4f}")
    print(f"New RMSE: {res_stretch.fun:.4f}")
    
    # Check extremes with best linear
    best_calibrated = res.x[0] * y_pred + res.x[1]
    
    df['calibrated'] = best_calibrated
    
    def get_grade(s):
        if s >= 90: return 'A'
        elif s < 60: return 'F'
        else: return 'Other'
        
    df['Grade'] = y_true.apply(get_grade)
    df['AbsError_Orig'] = (y_true - y_pred).abs()
    df['AbsError_Calib'] = (y_true - df['calibrated']).abs()
    
    print("\nImpact on Extremes (MAE):")
    print(df.groupby('Grade')[['AbsError_Orig', 'AbsError_Calib']].mean())

if __name__ == "__main__":
    optimize_calibration()
