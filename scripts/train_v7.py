import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data, RANDOM_STATE

def preprocess_v7(train, test):
    print("Feature Engineering (Phase 7 - Conservative)...")
    
    # Combine for operations
    train['is_train'] = 1
    test['is_train'] = 0
    all_data = pd.concat([train.drop(columns=['exam_score']), test], axis=0).reset_index(drop=True)
    
    # --- 1. Ordinal Encoding (Safe) ---
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_map = {'low': 0, 'medium': 1, 'high': 2}
    difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    
    all_data['sleep_quality_num'] = all_data['sleep_quality'].map(sleep_map)
    all_data['facility_rating_num'] = all_data['facility_rating'].map(facility_map)
    all_data['exam_difficulty_num'] = all_data['exam_difficulty'].map(difficulty_map)
    
    # --- 2. User Suggested Interactions ---
    
    # "Sleep Score": Quality * Duration
    all_data['sleep_score'] = all_data['sleep_hours'] * (1 + all_data['sleep_quality_num']) 
    # (Using 1+ to differentiate poor(0) from actual 0 hours, though sleep_hours > 4)
    
    # "Facility relative to Difficulty": Better facilities might mitigate hard exams?
    # Add epsilon or +1 to avoid division by zero if difficulty was 0 (it is 0,1,2)
    all_data['facility_diff_ratio'] = all_data['facility_rating_num'] / (all_data['exam_difficulty_num'] + 1)
    
    # --- 3. The "Proven" Winner ---
    all_data['study_attendance_interaction'] = all_data['study_hours'] * all_data['class_attendance']
    
    # --- 4. One-Hot Encoding (Standard) ---
    # Dropping Target Encoding/Clustering to avoid overfitting seen in V2
    categorical_cols = ['gender', 'course', 'internet_access', 'study_method']
    all_data = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)
    
    # Drop original ordinal/unused columns BUT keep is_train for splitting
    drop_cols = ['id', 'sleep_quality', 'facility_rating', 'exam_difficulty']
    all_data.drop(columns=drop_cols, inplace=True)
    
    # Split back
    train_final = all_data[all_data['is_train'] == 1].copy()
    test_final = all_data[all_data['is_train'] == 0].copy()
    
    # Now drop is_train
    train_final.drop(columns=['is_train'], inplace=True)
    test_final.drop(columns=['is_train'], inplace=True)
    
    # Add target back
    train_final['exam_score'] = train['exam_score'].values
    
    return train_final, test_final

def train_v7():
    train_df, test_df, submission = load_data()
    train_final, test_final = preprocess_v7(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final
    
    print(f"Features ({len(X.columns)}): {list(X.columns)}")
    
    # Standard XGB params to benchmark features purely
    model_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    importances = pd.Series(0.0, index=X.columns)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_preds = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_preds
        test_preds += np.clip(model.predict(X_test), 0, 100) / 5
        importances += model.feature_importances_ / 5
        
        print(f"Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y_val, val_preds)):.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nV7 Overall CV RMSE: {overall_rmse:.4f}")
    
    print("\n--- Top 10 Features ---")
    print(importances.sort_values(ascending=False).head(10))
    
    # Save submission
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_path = os.path.join(base_dir, 'submissions', 'submission_v7.csv')
    submission['exam_score'] = test_preds
    submission.to_csv(sub_path, index=False)
    print(f"V7 Submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_v7()
