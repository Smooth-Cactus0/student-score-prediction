import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Import Base Preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data, RANDOM_STATE

def preprocess_v20_features(train, test):
    print("Feature Engineering (Phase 20 - Meta Teacher)...")
    
    # Combine
    train['is_train'] = 1
    test['is_train'] = 0
    all_data = pd.concat([train.drop(columns=['exam_score']), test], axis=0).reset_index(drop=True)
    
    # --- 1. Base Encodings ---
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_map = {'low': 0, 'medium': 1, 'high': 2}
    difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    
    all_data['sleep_quality_num'] = all_data['sleep_quality'].map(sleep_map)
    all_data['facility_rating_num'] = all_data['facility_rating'].map(facility_map)
    all_data['exam_difficulty_num'] = all_data['exam_difficulty'].map(difficulty_map)
    
    # --- 2. Advanced Features ---
    
    # Bins
    all_data['attendance_bin'] = pd.cut(
        all_data['class_attendance'], 
        bins=[-1, 70, 90, 101], 
        labels=[0, 1, 2]
    ).astype(int)
    
    all_data['study_bin'] = pd.cut(
        all_data['study_hours'], 
        bins=[-1, 3, 7, 100], 
        labels=[0, 1, 2]
    ).astype(int)

    # Coaching
    all_data['is_coached'] = (all_data['study_method'] == 'coaching').astype(int)
    
    # Interactions
    all_data['dedication_score'] = (
        all_data['study_hours'] * 
        all_data['class_attendance'] * 
        (1 + all_data['sleep_quality_num'])
    )
    
    all_data['slacker_score'] = (1 / (all_data['study_hours'] + 1)) * (100 - all_data['class_attendance'])

    # New V20 Interactions
    all_data['study_difficulty'] = all_data['study_hours'] * (all_data['exam_difficulty_num'] + 1)
    all_data['attendance_facility'] = all_data['class_attendance'] * (all_data['facility_rating_num'] + 1)
    
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

def augment_data(X, y):
    # Identify Extremes for Augmentation (v16 strategy)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    high_mask = y > 85
    low_mask = y < 60
    
    X_high = X[high_mask].copy()
    y_high = y[high_mask].copy()
    
    X_low = X[low_mask].copy()
    y_low = y[low_mask].copy()
    
    # Noise injection
    noise_level = 0.05
    
    def add_noise(df):
        noisy = df.copy()
        for col in noisy.columns:
            if col in ['study_hours', 'class_attendance', 'sleep_hours', 'slacker_score', 'dedication_score', 'study_difficulty', 'attendance_facility']:
                 sigma = noisy[col].std() * noise_level
                 noise = np.random.normal(0, sigma, size=len(noisy))
                 noisy[col] += noise
        return noisy

    X_high_aug = add_noise(X_high)
    X_low_aug = add_noise(X_low)
    
    X_aug = pd.concat([X, X_high_aug, X_low_aug], axis=0)
    y_aug = pd.concat([y, y_high, y_low], axis=0)
    
    return X_aug, y_aug

def train_v20_meta_teacher():
    print("Loading Data for V20 Meta Teacher...")
    train_df, test_df, submission = load_data()
    
    # 1. Load Original Data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orig_path = os.path.join(base_dir, 'archive', 'Exam_Score_Prediction.csv')
    orig_df = pd.read_csv(orig_path)
    orig_df.rename(columns={'student_id': 'id'}, inplace=True)
    
    # 2. Load Pseudo Labels (Source: V19 Stacking)
    pseudo_path = os.path.join(base_dir, 'submissions', 'submission_v19_stacking.csv')
    if not os.path.exists(pseudo_path):
        print("V19 Stacking submission not found. Please run train_v19_stacking.py first.")
        return
    pseudo_df = pd.read_csv(pseudo_path)
    
    print(f"Original: {len(orig_df)}, Pseudo (V19): {len(pseudo_df)}")
    
    # 3. Preprocess Everything
    # We need to preprocess Train, Original, and Test
    # Let's combine Train and Original for the 'train' part of preprocess_v20
    train_combined = pd.concat([train_df, orig_df], axis=0).reset_index(drop=True)
    
    train_proc, test_proc = preprocess_v20_features(train_combined, test_df)
    
    # Recover splits
    n_train = len(train_df)
    n_orig = len(orig_df)
    
    X_train_only = train_proc.iloc[:n_train].drop(columns=['exam_score'])
    y_train_only = train_proc.iloc[:n_train]['exam_score']
    
    X_orig = train_proc.iloc[n_train:].drop(columns=['exam_score'])
    y_orig = train_proc.iloc[n_train:]['exam_score']
    
    X_test_pseudo = test_proc.copy()
    y_test_pseudo = pseudo_df['exam_score']
    
    # 4. Training Loop
    model_params = {
        'n_estimators': 2000,
        'learning_rate': 0.04,
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
    oof_preds = np.zeros(len(X_train_only))
    test_preds = np.zeros(len(X_test_pseudo))
    
    print("\nStarting V20 Meta Teacher Training...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_only, y_train_only)):
        # Validation
        X_val = X_train_only.iloc[val_idx]
        y_val = y_train_only.iloc[val_idx]
        
        # Training Set: Fold Train + Original + Pseudo
        X_tr_fold = X_train_only.iloc[train_idx]
        y_tr_fold = y_train_only.iloc[train_idx]
        
        # Combine
        X_train_full = pd.concat([X_tr_fold, X_orig, X_test_pseudo], axis=0)
        y_train_full = pd.concat([y_tr_fold, y_orig, y_test_pseudo], axis=0)
        
        # Augment ONLY the train fold + original part? 
        # Actually let's just augment the whole training pool for extremes.
        X_train_aug, y_train_aug = augment_data(X_train_full, y_train_full)
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(
            X_train_aug, y_train_aug,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_preds = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_preds
        test_preds += np.clip(model.predict(X_test_pseudo), 0, 100) / 5
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y_train_only, oof_preds))
    print(f"\nV20 Overall CV RMSE: {overall_rmse:.4f}")
    
    # Save OOF
    oof_df = train_df[['id']].copy()
    oof_df['exam_score'] = y_train_only
    oof_df['pred'] = oof_preds
    oof_path = os.path.join(base_dir, 'submissions', 'oof_v20_meta_teacher.csv')
    oof_df.to_csv(oof_path, index=False)
    
    # Save Submission
    submission['exam_score'] = test_preds
    sub_path = os.path.join(base_dir, 'submissions', 'submission_v20_meta_teacher.csv')
    submission.to_csv(sub_path, index=False)
    print(f"V20 Submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_v20_meta_teacher()
