import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_data, load_data, RANDOM_STATE

def preprocess_v6(train, test):
    # Start with V2 features
    train_final, test_final = preprocess_data(train, test)
    
    # Add V6 specific features
    print("Feature Engineering (Phase 6)...")
    
    # 1. Course Efficiency (Target dependent - need to be careful with leak)
    # Better: mean(exam_score) per course already captured by target encoding.
    # Let's try: study_hours normalized by course mean study_hours.
    # (Non-target dependent)
    course_study_mean = train.groupby('course')['study_hours'].mean().to_dict()
    train_final['course_study_rel'] = train['course'].map(course_study_mean) - train['study_hours']
    test_final['course_study_rel'] = test['course'].map(course_study_mean) - test['study_hours']
    
    # 2. Attendance x Difficulty
    train_final['attn_diff_inter'] = train_final['class_attendance'] * train_final['exam_difficulty_num']
    test_final['attn_diff_inter'] = test_final['class_attendance'] * test_final['exam_difficulty_num']
    
    # 3. Facility x Study Method
    # study_method_target is already in train_final
    train_final['fac_method_inter'] = train_final['facility_rating_num'] * train_final['study_method_target']
    test_final['fac_method_inter'] = test_final['facility_rating_num'] * test_final['study_method_target']
    
    return train_final, test_final

def train_v6():
    train_df, test_df, submission = load_data()
    train_final, test_final = preprocess_v6(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final
    
    # Use best params from Optuna (V5)
    best_params = {
        'learning_rate': 0.035,
        'max_depth': 8,
        'subsample': 0.95,
        'colsample_bytree': 0.77,
        'min_child_weight': 8,
        'reg_alpha': 7.5,
        'reg_lambda': 1.2,
        'n_estimators': 1000,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_preds = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_preds
        test_preds += np.clip(model.predict(X_test), 0, 100) / 5
        print(f"Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y_val, val_preds)):.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nV6 Overall CV RMSE: {overall_rmse:.4f}")
    
    # Save submission
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_path = os.path.join(base_dir, 'submissions', 'submission_v6.csv')
    submission['exam_score'] = test_preds
    submission.to_csv(sub_path, index=False)
    print(f"V6 Submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_v6()
