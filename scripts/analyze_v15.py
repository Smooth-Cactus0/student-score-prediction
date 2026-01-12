import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, confusion_matrix
import os
import sys

def get_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'

def analyze_v15_results():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof_path = os.path.join(base_dir, 'submissions', 'oof_v15_weighted.csv')
    
    if not os.path.exists(oof_path):
        print(f"Error: {oof_path} not found.")
        return
        
    print(f"Loading OOF predictions from {oof_path}...")
    df = pd.read_csv(oof_path)
    
    y = df['exam_score']
    oof_preds = df['pred']
    
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nOverall RMSE: {rmse:.4f}")

    # Grade Analysis
    df['Actual_Grade'] = y.apply(get_grade)
    df['Pred_Grade'] = oof_preds.apply(get_grade)
    df['AbsError'] = np.abs(y - oof_preds)
    
    print("\nMean Absolute Error by Grade:")
    print(df.groupby('Actual_Grade')['AbsError'].mean().sort_index())
    
    # Comparison to previous baseline (hardcoded from memory/previous logs)
    # Previous: A=8.61, F=7.09
    
    mae_a = df[df['Actual_Grade'] == 'A']['AbsError'].mean()
    mae_f = df[df['Actual_Grade'] == 'F']['AbsError'].mean()
    
    print(f"\nGrade A MAE: {mae_a:.4f} (Baseline: ~8.61)")
    print(f"Grade F MAE: {mae_f:.4f} (Baseline: ~7.09)")
    
    if mae_a < 8.61 or mae_f < 7.09:
        print("\nSUCCESS: Weighted training improved tails!")
    else:
        print("\nFAILURE: Weighted training did not improve tails.")

if __name__ == "__main__":
    analyze_v15_results()
