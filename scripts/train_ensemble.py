import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os

# Import our custom preprocessing from the sibling script
# We need to make sure the path is correct or module is found.
# Since we are running as a script, relative imports are tricky.
# We will use the direct file execution method or sys.path append.
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_data, load_data, RANDOM_STATE

def train_ensemble():
    print("Loading and Preprocessing...")
    train_df, test_df, submission = load_data()
    
    # Use the shared preprocessing pipeline
    train_final, test_final = preprocess_data(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final # No target here
    
    # Feature cleanup for models that might struggle with special characters (LightGBM sometimes)
    # But usually fine.
    
    print(f"\nFeature Count: {len(X.columns)}")

    # --- Modeling Setup ---
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Placeholders for OOF and Test predictions
    oof_xgb = np.zeros(len(X))
    oof_lgb = np.zeros(len(X))
    oof_hgb = np.zeros(len(X))
    
    test_xgb = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_hgb = np.zeros(len(X_test))
    
    # --- Model 1: XGBoost ---
    print("\n--- Training XGBoost ---")
    xgb_params = {
        'n_estimators': 800,
        'learning_rate': 0.04, # Increased slightly to compensate
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    }
    
    # --- Model 2: LightGBM ---
    print("--- Training LightGBM ---")
    lgb_params = {
        'n_estimators': 800,
        'learning_rate': 0.04,
        'max_depth': -1,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'objective': 'regression',
        'metric': 'rmse',
        'random_state': RANDOM_STATE,
        'verbosity': -1
    }
    
    # --- Model 3: HistGradientBoosting (Sklearn) ---
    print("--- Training HistGradientBoosting ---")
    hgb_params = {
        'max_iter': 800,
        'learning_rate': 0.04,
        'max_depth': 6,
        'random_state': RANDOM_STATE,
        'early_stopping': True,
        'validation_fraction': 0.1, # HGB uses internal validation usually, but we do manual CV loop
        'n_iter_no_change': 50
    }

    # CV Loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 1. XGBoost
        model_xgb = xgb.XGBRegressor(**xgb_params)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_pred = np.clip(model_xgb.predict(X_val), 0, 100)
        oof_xgb[val_idx] = val_pred
        test_xgb += np.clip(model_xgb.predict(X_test), 0, 100) / 5
        
        # 2. LightGBM
        model_lgb = lgb.LGBMRegressor(**lgb_params)
        # LGBM uses 'eval_set' and 'eval_metric'
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=callbacks)
        val_pred = np.clip(model_lgb.predict(X_val), 0, 100)
        oof_lgb[val_idx] = val_pred
        test_lgb += np.clip(model_lgb.predict(X_test), 0, 100) / 5
        
        # 3. HistGradientBoosting
        # HGB doesn't support 'eval_set' in fit() the same way for early stopping easily without internal split.
        # We will trust its internal validation or just run fixed iters if strictly following CV.
        # To enable early stopping properly with OUR val set, we can't easily do it in sklearn API without passing the set.
        # But HGB allows passing sample_weight etc.
        # Actually, HGB has 'early_stopping=True' which splits X_train internally. 
        # For this exercise, we'll just let it run or rely on internal split (default 0.1).
        model_hgb = HistGradientBoostingRegressor(**hgb_params)
        model_hgb.fit(X_train, y_train) 
        val_pred = np.clip(model_hgb.predict(X_val), 0, 100)
        oof_hgb[val_idx] = val_pred
        test_hgb += np.clip(model_hgb.predict(X_test), 0, 100) / 5
        
        print(f"Fold {fold+1} Completed")

    # Metrics
    rmse_xgb = np.sqrt(mean_squared_error(y, oof_xgb))
    rmse_lgb = np.sqrt(mean_squared_error(y, oof_lgb))
    rmse_hgb = np.sqrt(mean_squared_error(y, oof_hgb))
    
    print(f"\n--- Model Performance (RMSE) ---")
    print(f"XGBoost: {rmse_xgb:.4f}")
    print(f"LightGBM: {rmse_lgb:.4f}")
    print(f"HistGB:  {rmse_hgb:.4f}")
    
    # --- Ensembling (Simple Average) ---
    oof_ensemble = (oof_xgb + oof_lgb + oof_hgb) / 3
    rmse_ensemble = np.sqrt(mean_squared_error(y, oof_ensemble))
    
    print(f"\nEnsemble RMSE: {rmse_ensemble:.4f}")
    
    # --- Submission ---
    test_ensemble = (test_xgb + test_lgb + test_hgb) / 3
    submission['exam_score'] = np.clip(test_ensemble, 0, 100)
    
    # Robust path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    os.makedirs(sub_dir, exist_ok=True)
    
    save_path = os.path.join(sub_dir, 'submission_ensemble_v3.csv')
    submission.to_csv(save_path, index=False)
    print(f"\nSubmission file saved to '{save_path}'")

if __name__ == "__main__":
    train_ensemble()
