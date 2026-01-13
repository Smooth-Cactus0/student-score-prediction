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

def preprocess_v24(train, test):
    print("Feature Engineering (Phase 24 - Feature Boost)...")
    
    # Combine for consistent encoding
    train['is_train'] = 1
    test['is_train'] = 0
    all_data = pd.concat([train.drop(columns=['exam_score']), test], axis=0).reset_index(drop=True)
    
    # --- 1. Base Encodings (from V7/Baseline) ---
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    all_data['sleep_quality_num'] = all_data['sleep_quality'].map(sleep_map)
    
    # Ordinal Encoding for others
    facility_map = {'low': 0, 'medium': 1, 'high': 2}
    difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    all_data['facility_rating_num'] = all_data['facility_rating'].map(facility_map)
    all_data['exam_difficulty_num'] = all_data['exam_difficulty'].map(difficulty_map)
    
    # --- 2. The "Winner" Features from Discovery ---
    
    # From Phase 3 (Best for Grade A)
    all_data['dedication'] = all_data['study_hours'] * all_data['class_attendance']
    all_data['dedication_sleep'] = all_data['dedication'] * (all_data['sleep_quality_num'] + 1)
    all_data['attend_sq'] = all_data['class_attendance'] ** 2
    
    # From Phase 2 (Best for Global/F)
    all_data['flag_low_attend'] = (all_data['class_attendance'] < 70).astype(int)
    all_data['flag_low_study'] = (all_data['study_hours'] < 3).astype(int)
    all_data['flag_high_performer'] = ((all_data['study_hours'] > 7) & (all_data['class_attendance'] > 90)).astype(int)
    
    # From Phase 5 (Psychology - reduced set)
    # The "Slacker" (Smart but lazy?)
    all_data['type_slacker'] = ((all_data['class_attendance'] > 80) & (all_data['study_hours'] < 2)).astype(int)
    
    # --- 3. Standard One-Hot ---
    cat_cols = ['gender', 'course', 'study_method', 'internet_access']
    all_data = pd.get_dummies(all_data, columns=cat_cols, drop_first=True)
    
    # Split
    train_final = all_data[all_data['is_train'] == 1].copy()
    test_final = all_data[all_data['is_train'] == 0].copy()
    
    # Drop unused originals
    drop_cols = ['id', 'is_train', 'sleep_quality', 'facility_rating', 'exam_difficulty']
    train_final.drop(columns=drop_cols, inplace=True)
    test_final.drop(columns=drop_cols, inplace=True)
    
    train_final['exam_score'] = train['exam_score'].values
    
    return train_final, test_final

def train_v24():
    print("Loading Data for V24...")
    train_df, test_df, submission = load_data()
    
    # 1. Load Original Data (Critical for best performance)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orig_path = os.path.join(base_dir, 'archive', 'Exam_Score_Prediction.csv')
    orig_df = pd.read_csv(orig_path)
    orig_df.rename(columns={'student_id': 'id'}, inplace=True)
    
    # 2. Pseudo Labels (Use V23 Hybrid as it's the best current estimator)
    # Using V23 instead of V19/V7 might help propogate the "Extreme" corrections
    pseudo_path = os.path.join(base_dir, 'submissions', 'submission_v23_hybrid.csv')
    if not os.path.exists(pseudo_path):
        print("V23 submission not found. Using V21.")
        pseudo_path = os.path.join(base_dir, 'submissions', 'submission_v21_mega_stacking.csv')
        
    pseudo_df = pd.read_csv(pseudo_path)
    print(f"Using Pseudo Labels from: {os.path.basename(pseudo_path)}")
    
    # 3. Preprocess Everything
    # Combine Train+Orig for fitting encoders/stats
    train_combined = pd.concat([train_df, orig_df], axis=0).reset_index(drop=True)
    
    train_proc, test_proc = preprocess_v24(train_combined, test_df)
    
    # Recover splits
    n_train = len(train_df)
    X_train_only = train_proc.iloc[:n_train].drop(columns=['exam_score'])
    y_train_only = train_proc.iloc[:n_train]['exam_score']
    
    X_orig = train_proc.iloc[n_train:].drop(columns=['exam_score'])
    y_orig = train_proc.iloc[n_train:]['exam_score']
    
    X_test_pseudo = test_proc.copy()
    y_test_pseudo = pseudo_df['exam_score']
    
    # 4. Training (XGBoost)
    model_params = {
        'n_estimators': 2000,
        'learning_rate': 0.04,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X_train_only))
    test_preds = np.zeros(len(X_test_pseudo))
    
    print("\nStarting V24 Feature Boost Training...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_only, y_train_only)):
        X_val = X_train_only.iloc[val_idx]
        y_val = y_train_only.iloc[val_idx]
        
        # Train on: Fold Train + Original + Pseudo
        X_tr_fold = X_train_only.iloc[train_idx]
        y_tr_fold = y_train_only.iloc[train_idx]
        
        X_train_full = pd.concat([X_tr_fold, X_orig, X_test_pseudo], axis=0)
        y_train_full = pd.concat([y_tr_fold, y_orig, y_test_pseudo], axis=0)
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(
            X_train_full, y_train_full,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_pred = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_pred
        test_preds += np.clip(model.predict(X_test_pseudo), 0, 100) / 5
        
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y_train_only, oof_preds))
    print(f"\nV24 Overall CV RMSE: {overall_rmse:.4f}")
    
    # Analyze Grade A/F locally
    df_eval = pd.DataFrame({'true': y_train_only, 'pred': oof_preds})
    df_eval['abs_err'] = (df_eval['true'] - df_eval['pred']).abs()
    mae_a = df_eval[df_eval['true'] >= 90]['abs_err'].mean()
    mae_f = df_eval[df_eval['true'] < 60]['abs_err'].mean()
    print(f"V24 Grade A MAE: {mae_a:.4f} (Baseline ~8.52)")
    print(f"V24 Grade F MAE: {mae_f:.4f} (Baseline ~7.05)")
    
    # Save
    sub_dir = os.path.join(base_dir, 'submissions')
    submission['exam_score'] = test_preds
    sub_path = os.path.join(sub_dir, 'submission_v24_feature_boost.csv')
    submission.to_csv(sub_path, index=False)
    
    oof_df = train_df[['id']].copy()
    oof_df['exam_score'] = y_train_only
    oof_df['pred'] = oof_preds
    oof_df.to_csv(os.path.join(sub_dir, 'oof_v24_feature_boost.csv'), index=False)
    print(f"Submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_v24()
