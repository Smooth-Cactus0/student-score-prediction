import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

def preprocess_data(train, test):
    """
    Applies Phase 2 Feature Engineering.
    """
    print("Feature Engineering (Shared Pipeline)...")
    
    # Combine for non-target dependent operations
    train['is_train'] = 1
    test['is_train'] = 0
    all_data = pd.concat([train.drop(columns=['exam_score']), test], axis=0).reset_index(drop=True)
    
    # --- 1. Ordinal Encoding ---
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_map = {'low': 0, 'medium': 1, 'high': 2}
    difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    
    all_data['sleep_quality_num'] = all_data['sleep_quality'].map(sleep_map)
    all_data['facility_rating_num'] = all_data['facility_rating'].map(facility_map)
    all_data['exam_difficulty_num'] = all_data['exam_difficulty'].map(difficulty_map)
    
    # --- 2. Advanced Interactions ---
    all_data['study_attendance_interaction'] = all_data['study_hours'] * all_data['class_attendance']
    all_data['study_attendance_squared'] = all_data['study_attendance_interaction'] ** 2
    all_data['study_attendance_log'] = np.log1p(all_data['study_attendance_interaction'])
    all_data['study_per_sleep'] = all_data['study_hours'] / (all_data['sleep_hours'] + 0.1)
    
    # --- 3. Clustering ---
    cluster_cols = ['study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality_num']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_data[cluster_cols])
    
    kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
    all_data['student_cluster'] = kmeans.fit_predict(scaled_features)
    
    # --- 4. Target Encoding ---
    train_processed = all_data[all_data['is_train'] == 1].copy()
    test_processed = all_data[all_data['is_train'] == 0].copy()
    
    train_processed['exam_score'] = train['exam_score'].values
    
    for col in ['course', 'study_method']:
        tr_enc, te_enc = target_encode(train_processed, test_processed, col, 'exam_score')
        train_processed[f'{col}_target'] = tr_enc
        test_processed[f'{col}_target'] = te_enc
        
    dummy_cols = ['gender', 'internet_access', 'student_cluster']
    
    train_dummies = pd.get_dummies(train_processed[dummy_cols], columns=dummy_cols, drop_first=True)
    test_dummies = pd.get_dummies(test_processed[dummy_cols], columns=dummy_cols, drop_first=True)
    
    train_dummies, test_dummies = train_dummies.align(test_dummies, join='left', axis=1, fill_value=0)
    
    drop_cols = ['id', 'is_train', 'sleep_quality', 'facility_rating', 'exam_difficulty', 
                 'course', 'study_method', 'gender', 'internet_access', 'student_cluster']
    
    train_final = pd.concat([train_processed.drop(columns=drop_cols), train_dummies], axis=1)
    test_final = pd.concat([test_processed.drop(columns=drop_cols), test_dummies], axis=1)
    
    return train_final, test_final
