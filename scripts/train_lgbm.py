import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Import V7 preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_v7 import preprocess_v7
from preprocessing import load_data, RANDOM_STATE

def train_lgbm():
    print("Loading Data for LightGBM...")
    train_df, test_df, submission = load_data()
    
    # Use V7 Conservative Preprocessing (Proven best)
    print("Preprocessing (V7 Pipeline)...")
    train_final, test_final = preprocess_v7(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final
    
    # LightGBM Params - Diverse from XGBoost
    # Using 'gbdt' (Gradient Boosting Decision Tree)
    model_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'num_leaves': 31,           # LGBM specific, controls complexity (~ depth=5)
        'max_depth': -1,            # Let num_leaves control depth
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
        'verbosity': -1
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    print("\nStarting LightGBM Training...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # LightGBM Scikit-Learn API
        model = lgb.LGBMRegressor(**model_params)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0) # Silence
        ]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=callbacks
        )
        
        val_preds = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_preds
        test_preds += np.clip(model.predict(X_test), 0, 100) / 5
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nLightGBM Overall CV RMSE: {overall_rmse:.4f}")
    
    # Save OOF
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof_df = train_df[['id']].copy()
    oof_df['exam_score'] = y
    oof_df['pred'] = oof_preds
    oof_path = os.path.join(base_dir, 'submissions', 'oof_lgbm.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to '{oof_path}'")
    
    # Save submission
    sub_path = os.path.join(base_dir, 'submissions', 'submission_lgbm.csv')
    submission['exam_score'] = test_preds
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_lgbm()
