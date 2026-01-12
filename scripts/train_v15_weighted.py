import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Import V7 preprocessing for base, but we will extend it
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data, RANDOM_STATE

def preprocess_v15_features(train, test):
    print("Feature Engineering (Phase 15 - Extreme Focused)...")
    
    # Combine
    train['is_train'] = 1
    test['is_train'] = 0
    all_data = pd.concat([train.drop(columns=['exam_score']), test], axis=0).reset_index(drop=True)
    
    # --- 1. Base Encodings (from V7) ---
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_map = {'low': 0, 'medium': 1, 'high': 2}
    difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    
    all_data['sleep_quality_num'] = all_data['sleep_quality'].map(sleep_map)
    all_data['facility_rating_num'] = all_data['facility_rating'].map(facility_map)
    all_data['exam_difficulty_num'] = all_data['exam_difficulty'].map(difficulty_map)
    
    # --- 2. New "Extreme" Features ---
    
    # A. Discrete Bins (Trees love these for splits)
    # Binning Attendance: <70 (Fail Zone), 70-90 (Mid), >90 (A Zone)
    all_data['attendance_bin'] = pd.cut(
        all_data['class_attendance'], 
        bins=[-1, 70, 90, 101], 
        labels=[0, 1, 2]
    ).astype(int)
    
    # Binning Study Hours: <3 (Low), 3-7 (Mid), >7 (High)
    all_data['study_bin'] = pd.cut(
        all_data['study_hours'], 
        bins=[-1, 3, 7, 100], 
        labels=[0, 1, 2]
    ).astype(int)

    # B. "Coaching" Flag (Strong signal for A students)
    all_data['is_coached'] = (all_data['study_method'] == 'coaching').astype(int)
    
    # C. Interaction: The "Genius" Interaction
    # High Study + High Attendance + Good Sleep
    all_data['dedication_score'] = (
        all_data['study_hours'] * 
        all_data['class_attendance'] * 
        (1 + all_data['sleep_quality_num'])
    )
    
    # D. "Lazy" Interaction
    # Low study * Low attendance
    all_data['slacker_score'] = (1 / (all_data['study_hours'] + 1)) * (100 - all_data['class_attendance'])

    # --- 3. One-Hot Encoding ---
    categorical_cols = ['gender', 'course', 'internet_access', 'study_method']
    all_data = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)
    
    # Drop originals
    drop_cols = ['id', 'sleep_quality', 'facility_rating', 'exam_difficulty']
    all_data.drop(columns=drop_cols, inplace=True)
    
    # Split
    train_final = all_data[all_data['is_train'] == 1].copy()
    test_final = all_data[all_data['is_train'] == 0].copy()
    
    train_final.drop(columns=['is_train'], inplace=True)
    test_final.drop(columns=['is_train'], inplace=True)
    
    train_final['exam_score'] = train['exam_score'].values
    
    return train_final, test_final

def train_v15_weighted():
    print("Loading Data for V15 Weighted Training...")
    train_df, test_df, submission = load_data()
    
    train_final, test_final = preprocess_v15_features(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final
    
    # --- Sample Weights Calculation ---
    # Goal: Give higher weight to predictions at the tails ( <60 and >85 )
    # This counteracts the "regress to mean" tendency.
    
    # Strategy: Parabolic weighting centered at mean
    mu = y.mean()
    # Normalize deviation
    deviation = np.abs(y - mu)
    
    # Weights: 1.0 (base) + scaling * deviation
    # We want weights to range roughly from 1.0 (mean) to ~3.0 (extremes)
    # Max deviation is approx 40 (100-60). 
    # Let's try: weight = 1 + (deviation / 20)^2
    sample_weights = 1.0 + (deviation / 20.0) ** 2
    
    print(f"Sample Weights Stats: Min={sample_weights.min():.2f}, Mean={sample_weights.mean():.2f}, Max={sample_weights.max():.2f}")
    
    # XGBoost Params
    model_params = {
        'n_estimators': 2000,
        'learning_rate': 0.04, # Slightly lower LR for stability with weights
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    print("\nStarting Weighted Training...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = sample_weights.iloc[train_idx]
        
        # Note: We do NOT pass sample weights to eval_set, 
        # because we want early stopping based on REAL RMSE, not weighted RMSE.
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_preds = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_preds
        test_preds += np.clip(model.predict(X_test), 0, 100) / 5
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nV15 Overall CV RMSE: {overall_rmse:.4f}")
    
    # Save submission
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Save OOF for analysis
    oof_df = train_df[['id']].copy()
    oof_df['exam_score'] = y
    oof_df['pred'] = oof_preds
    oof_path = os.path.join(base_dir, 'submissions', 'oof_v15_weighted.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to '{oof_path}'")
    
    sub_path = os.path.join(base_dir, 'submissions', 'submission_v15_weighted.csv')
    submission['exam_score'] = test_preds
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_v15_weighted()
