import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, confusion_matrix
import os
import sys

# Import V7 preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_v7 import preprocess_v7
from preprocessing import load_data, RANDOM_STATE

def get_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'

def analyze_performance():
    print("Loading Data for Analysis...")
    train_df, test_df, _ = load_data()
    train_final, _ = preprocess_v7(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    
    # Using LightGBM as our Representative Model
    # (Fast to train, similar performance to XGBoost)
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
        'verbosity': -1
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    
    print("Generating OOF Predictions (Re-training LightGBM)...")
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**model_params)
        model.fit(X_train, y_train)
        oof_preds[val_idx] = np.clip(model.predict(X_val), 0, 100)

    # --- Analysis ---
    residuals = y - oof_preds
    mse = mean_squared_error(y, oof_preds)
    rmse = np.sqrt(mse)
    print(f"\nOverall RMSE: {rmse:.4f}")

    # 1. Grade Confusion Matrix
    print("Generating Grade Confusion Matrix...")
    y_grades = y.apply(get_grade)
    pred_grades = pd.Series(oof_preds).apply(get_grade)
    
    labels = ['A', 'B', 'C', 'D', 'F']
    cm = confusion_matrix(y_grades, pred_grades, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Grade')
    plt.ylabel('Actual Grade')
    plt.title(f'Confusion Matrix (RMSE: {rmse:.4f})')
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, 'grade_confusion_matrix.png'))
    print(f"Saved: {os.path.join(img_dir, 'grade_confusion_matrix.png')}")

    # 2. Residual Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=oof_preds, y=residuals, alpha=0.3)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Score')
    plt.ylabel('Residual (Actual - Pred)')
    plt.title('Residuals vs Predictions')
    plt.savefig(os.path.join(img_dir, 'residual_analysis.png'))
    print(f"Saved: {os.path.join(img_dir, 'residual_analysis.png')}")

    # 3. Class-wise MAE
    df_analysis = pd.DataFrame({'Actual': y, 'Pred': oof_preds, 'Grade': y_grades})
    df_analysis['AbsError'] = np.abs(df_analysis['Actual'] - df_analysis['Pred'])
    
    print("\nMean Absolute Error by Grade:")
    print(df_analysis.groupby('Grade')['AbsError'].mean().sort_index())

if __name__ == "__main__":
    analyze_performance()
