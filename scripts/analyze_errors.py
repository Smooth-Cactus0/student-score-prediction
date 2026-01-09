import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_data, load_data, RANDOM_STATE

def analyze_errors():
    print("Loading and Preprocessing...")
    train_df, test_df, _ = load_data()
    train_final, _ = preprocess_data(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    
    # Use best params from V2/Optuna or just V2 defaults
    model_params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    
    print("Generating Out-of-Fold predictions...")
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)

    residuals = y - oof_preds
    
    # Robust path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    # 1. Residual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(oof_preds, residuals, alpha=0.1)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Score')
    plt.ylabel('Residual')
    plt.title('Residual Plot')
    plt.savefig(os.path.join(img_dir, 'residual_plot.png'))
    
    # 2. Distribution of Errors
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of Residuals')
    plt.savefig(os.path.join(img_dir, 'residuals_hist.png'))
    
    # 3. Error by Feature (Top Features)
    # Check if error is higher for certain study_hours
    analysis_df = train_df.copy()
    analysis_df['residual'] = residuals
    analysis_df['abs_error'] = np.abs(residuals)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=analysis_df, x='study_hours', y='abs_error')
    plt.title('Mean Absolute Error by Study Hours')
    plt.savefig(os.path.join(img_dir, 'error_by_study_hours.png'))

    print(f"\nError analysis complete. Images saved in '{img_dir}'")
    
    # Identify hardest samples
    hardest_samples = analysis_df.sort_values('abs_error', ascending=False).head(10)
    print("\n--- Top 10 Hardest Samples to Predict ---")
    print(hardest_samples[['study_hours', 'class_attendance', 'exam_score', 'residual']])

if __name__ == "__main__":
    analyze_errors()
