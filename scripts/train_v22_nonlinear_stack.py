import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def train_nonlinear_stack_v22():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    data_dir = os.path.join(base_dir, 'data')
    
    print("Loading Base Model OOFs...")
    # Base Models
    model_files = {
        'v13': 'oof_v13_teacher.csv',
        'lgbm': 'oof_lgbm.csv',
        'ann': 'oof_ann.csv',
        'hgb': 'oof_hgb.csv',
        'v16': 'oof_v16_aug.csv',
        'v20': 'oof_v20_meta_teacher.csv'
    }
    
    # Load Train Data for Original Features
    # The meta-learner needs context (e.g., is this a high study_hours student?)
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    # We need to make sure IDs match. OOFs usually have 'id'.
    # Let's align everything by 'id'.
    
    df_meta = train_df[['id', 'study_hours', 'class_attendance', 'sleep_hours', 'exam_score']].copy()
    
    for name, filename in model_files.items():
        path = os.path.join(sub_dir, filename)
        if os.path.exists(path):
            oof = pd.read_csv(path)
            # Merge on ID to be safe
            df_meta = df_meta.merge(oof[['id', 'pred']], on='id', suffixes=('', f'_{name}'))
            df_meta.rename(columns={'pred': f'pred_{name}'}, inplace=True)
        else:
            print(f"Warning: {filename} not found.")
            return

    # Drop ID and Target from X
    y = df_meta['exam_score']
    X = df_meta.drop(columns=['id', 'exam_score'])
    
    print(f"Meta-Features: {list(X.columns)}")
    
    # --- Test Meta-Features ---
    print("Preparing Test Meta-Features...")
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    df_test_meta = test_df[['id', 'study_hours', 'class_attendance', 'sleep_hours']].copy()
    
    sub_files = {
        'v13': 'submission_v13_teacher.csv',
        'lgbm': 'submission_lgbm.csv',
        'ann': 'submission_ann.csv',
        'hgb': 'submission_hgb.csv',
        'v16': 'submission_v16_aug.csv',
        'v20': 'submission_v20_meta_teacher.csv'
    }
    
    for name, filename in sub_files.items():
        path = os.path.join(sub_dir, filename)
        if os.path.exists(path):
            sub = pd.read_csv(path)
            # Assuming submissions are sorted by ID or we strictly merge?
            # Submissions don't always have ID in a way that matches perfectly if sorted differently.
            # But standard Kaggle submission is id sorted.
            # Let's simple assignment if lengths match
            df_test_meta[f'pred_{name}'] = sub['exam_score']
            
    X_test = df_test_meta.drop(columns=['id'])
    
    # --- Training XGBoost Meta-Learner ---
    # We want a shallow tree to prevent overfitting to the OOF noise
    meta_params = {
        'n_estimators': 200,
        'learning_rate': 0.03,
        'max_depth': 4,          # Shallow depth is key for stacking
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'early_stopping_rounds': 30
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    print("Training V22 Non-Linear Stacker (XGBoost)...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**meta_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test) / 5
        
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nV22 Non-Linear Stack Overall RMSE: {overall_rmse:.4f}")
    
    # Save OOF
    oof_df = df_meta[['id']].copy()
    oof_df['exam_score'] = y
    oof_df['pred'] = oof_preds
    oof_path = os.path.join(sub_dir, 'oof_v22_nonlinear.csv')
    oof_df.to_csv(oof_path, index=False)
    
    # Save Submission
    sub_sample = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    sub_sample['exam_score'] = np.clip(test_preds, 0, 100)
    sub_path = os.path.join(sub_dir, 'submission_v22_nonlinear.csv')
    sub_sample.to_csv(sub_path, index=False)
    print(f"Submission saved to '{sub_path}'")
    
    # Feature Importance
    print("\nMeta-Learner Feature Importance:")
    imps = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    print(imps.sort_values('Importance', ascending=False))

if __name__ == "__main__":
    train_nonlinear_stack_v22()
