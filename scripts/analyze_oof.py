import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error

def get_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'

def analyze_oof(oof_filename):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof_path = os.path.join(base_dir, 'submissions', oof_filename)
    
    if not os.path.exists(oof_path):
        print(f"Error: {oof_path} not found.")
        return
        
    print(f"Loading OOF predictions from {oof_filename}...")
    df = pd.read_csv(oof_path)
    
    y = df['exam_score']
    oof_preds = df['pred']
    
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nOverall RMSE: {rmse:.4f}")

    df['Actual_Grade'] = y.apply(get_grade)
    df['AbsError'] = np.abs(y - oof_preds)
    
    print("\nMean Absolute Error by Grade:")
    print(df.groupby('Actual_Grade')['AbsError'].mean().sort_index())
    
    mae_a = df[df['Actual_Grade'] == 'A']['AbsError'].mean()
    mae_f = df[df['Actual_Grade'] == 'F']['AbsError'].mean()
    
    print(f"\nGrade A MAE: {mae_a:.4f} (Baseline: ~8.61)")
    print(f"Grade F MAE: {mae_f:.4f} (Baseline: ~7.09)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_oof(sys.argv[1])
    else:
        analyze_oof("oof_v16_aug.csv")
