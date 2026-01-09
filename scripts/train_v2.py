import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random state for reproducibility
RANDOM_STATE = 42

def load_data():
    print("Loading data...")
    # Adjust paths for the new directory structure, robust to CWD
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'data')
    
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    return train, test, submission

def target_encode(train_df, test_df, col, target_col, smooth=10):
    """
    Applies Target Encoding with smoothing.
    """
    # Calculate global mean
    global_mean = train_df[target_col].mean()
    
    # Calculate aggregated statistics
    agg = train_df.groupby(col)[target_col].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    
    # Compute the "smoothed" means
    smooth_means = (counts * means + smooth * global_mean) / (counts + smooth)
    
    # Map to train and test
    train_encoded = train_df[col].map(smooth_means)
    test_encoded = test_df[col].map(smooth_means)
    
    # Fill missing values in test (if any category in test wasn't in train) with global mean
    test_encoded.fillna(global_mean, inplace=True)
    
    return train_encoded, test_encoded

def preprocess_data_v2(train, test):
    print("Feature Engineering (Phase 2)...")
    
    # Combine for non-target dependent operations
    train['is_train'] = 1
    test['is_train'] = 0
    all_data = pd.concat([train.drop(columns=['exam_score']), test], axis=0).reset_index(drop=True)
    
    # --- 1. Ordinal Encoding (Same as V1) ---
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_map = {'low': 0, 'medium': 1, 'high': 2}
    difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    
    all_data['sleep_quality_num'] = all_data['sleep_quality'].map(sleep_map)
    all_data['facility_rating_num'] = all_data['facility_rating'].map(facility_map)
    all_data['exam_difficulty_num'] = all_data['exam_difficulty'].map(difficulty_map)
    
    # --- 2. Advanced Interactions ---
    # Base interaction
    all_data['study_attendance_interaction'] = all_data['study_hours'] * all_data['class_attendance']
    
    # Polynomials
    all_data['study_attendance_squared'] = all_data['study_attendance_interaction'] ** 2
    all_data['study_attendance_log'] = np.log1p(all_data['study_attendance_interaction'])
    
    # Efficiency
    all_data['study_per_sleep'] = all_data['study_hours'] / (all_data['sleep_hours'] + 0.1)
    
    # --- 3. Clustering (Behavioral Profiles) ---
    # Features to cluster on
    cluster_cols = ['study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality_num']
    
    # Scale before clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_data[cluster_cols])
    
    kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
    all_data['student_cluster'] = kmeans.fit_predict(scaled_features)
    
    # --- 4. Target Encoding (replacing One-Hot for high cardinality/noisy cols) ---
    # Split back to apply target encoding (must only use Train info)
    train_processed = all_data[all_data['is_train'] == 1].copy()
    test_processed = all_data[all_data['is_train'] == 0].copy()
    
    # Add target back to train for encoding calculation
    train_processed['exam_score'] = train['exam_score'].values
    
    # Target Encode 'course' and 'study_method'
    for col in ['course', 'study_method']:
        tr_enc, te_enc = target_encode(train_processed, test_processed, col, 'exam_score')
        train_processed[f'{col}_target'] = tr_enc
        test_processed[f'{col}_target'] = te_enc
        
    # Still use One-Hot for 'gender' and 'internet_access' (low cardinality)
    # And maybe 'student_cluster' (categorical)
    dummy_cols = ['gender', 'internet_access', 'student_cluster']
    
    # Concat again to get_dummies consistently
    # (We need to be careful not to lose the target encoded cols)
    # Let's just do get_dummies on the specific columns and join them
    
    # Create dummies for both
    train_dummies = pd.get_dummies(train_processed[dummy_cols], columns=dummy_cols, drop_first=True)
    test_dummies = pd.get_dummies(test_processed[dummy_cols], columns=dummy_cols, drop_first=True)
    
    # Align columns (in case some cluster/category is missing in test)
    train_dummies, test_dummies = train_dummies.align(test_dummies, join='left', axis=1, fill_value=0)
    
    # Drop original categorical columns & helper columns
    drop_cols = ['id', 'is_train', 'sleep_quality', 'facility_rating', 'exam_difficulty', 
                 'course', 'study_method', 'gender', 'internet_access', 'student_cluster']
    
    train_final = pd.concat([train_processed.drop(columns=drop_cols), train_dummies], axis=1)
    test_final = pd.concat([test_processed.drop(columns=drop_cols), test_dummies], axis=1)
    
    return train_final, test_final

def train_and_evaluate_v2():
    train_df, test_df, submission = load_data()
    
    train_final, test_final = preprocess_data_v2(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final # No target here
    
    print(f"\nFinal Feature Count: {len(X.columns)}")
    print(f"Features: {list(X.columns)}")

    # --- Modeling (XGBoost) ---
    print("\nStarting Training (5-Fold CV)...")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    rmse_scores = []
    
    # Slightly updated params
    model_params = {
        'n_estimators': 800, # Reduced for speed
        'learning_rate': 0.04, # Increased slightly
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    }

    importances = pd.Series(0.0, index=X.columns)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**model_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_preds = model.predict(X_val)
        val_preds = np.clip(val_preds, 0, 100) # Validity clip
        oof_preds[val_idx] = val_preds
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        rmse_scores.append(rmse)
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")
        
        test_preds += model.predict(X_test) / 5
        importances += model.feature_importances_ / 5

    # Overall Metrics
    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nOverall CV RMSE: {overall_rmse:.4f}")
    print(f"Average Fold RMSE: {np.mean(rmse_scores):.4f}")

    # --- Feature Importance ---
    print("\n--- Feature Importance (Top 20) ---")
    print(importances.sort_values(ascending=False).head(20))

    # --- Submission ---
    submission['exam_score'] = np.clip(test_preds, 0, 100)
    
    # Ensure directory exists (robust path)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    os.makedirs(sub_dir, exist_ok=True)
    
    save_path = os.path.join(sub_dir, 'submission_v2.csv')
    submission.to_csv(save_path, index=False)
    print(f"\nSubmission file saved to '{save_path}'")

if __name__ == "__main__":
    train_and_evaluate_v2()
