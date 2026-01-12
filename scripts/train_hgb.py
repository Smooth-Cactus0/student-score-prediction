import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Import V7 preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_v7 import preprocess_v7
from preprocessing import load_data, RANDOM_STATE

def train_hgb():
    print("Loading Data for HistGradientBoosting (Sklearn)...")
    train_df, test_df, submission = load_data()
    
    print("Preprocessing (V7 Pipeline)...")
    train_final, test_final = preprocess_v7(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final
    
    # HistGradientBoostingRegressor Params
    # Similar to LightGBM/CatBoost
    model_params = {
        'learning_rate': 0.05,
        'max_iter': 2000,
        'max_depth': 6,
        'min_samples_leaf': 20,
        'l2_regularization': 0.1,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 50,
        'random_state': RANDOM_STATE,
        'verbose': 0
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    print("\nStarting HGB Training...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = HistGradientBoostingRegressor(**model_params)
        model.fit(X_train, y_train)
        
        val_preds = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_preds
        test_preds += np.clip(model.predict(X_test), 0, 100) / 5
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nHGB Overall CV RMSE: {overall_rmse:.4f}")
    
    # Save OOF
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof_df = train_df[['id']].copy()
    oof_df['exam_score'] = y
    oof_df['pred'] = oof_preds
    oof_path = os.path.join(base_dir, 'submissions', 'oof_hgb.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to '{oof_path}'")
    
    # Save submission
    sub_path = os.path.join(base_dir, 'submissions', 'submission_hgb.csv')
    submission['exam_score'] = test_preds
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_hgb()
