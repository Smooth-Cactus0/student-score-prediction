import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Import V15 preprocessing (it had good features, let's keep them)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_v15_weighted import preprocess_v15_features
from preprocessing import load_data, RANDOM_STATE

def augment_data(X, y):
    # Identify Extremes
    # High: > 85, Low: < 60
    high_mask = y > 85
    low_mask = y < 60
    
    X_high = X[high_mask].copy()
    y_high = y[high_mask].copy()
    
    X_low = X[low_mask].copy()
    y_low = y[low_mask].copy()
    
    print(f"Augmenting: {len(X_high)} High samples, {len(X_low)} Low samples")
    
    # Noise injection parameters
    noise_level = 0.05
    
    def add_noise(df):
        noisy = df.copy()
        # Add noise only to continuous features that matter
        # study_hours, class_attendance, sleep_hours (if exists)
        # Note: preprocess_v15 removed sleep_hours (used in interaction).
        # Let's check columns.
        cols_to_noise = ['study_hours', 'class_attendance'] # These might be gone if binned?
        # Check if they exist in processed df. 
        # Actually preprocess_v15 keeps them if we didn't drop them.
        # Looking at preprocess_v15: it drops 'id', 'sleep_quality'...
        # It does NOT drop 'study_hours' or 'class_attendance'.
        
        for col in noisy.columns:
            if col in ['study_hours', 'class_attendance', 'sleep_hours', 'slacker_score', 'dedication_score']:
                 sigma = noisy[col].std() * noise_level
                 noise = np.random.normal(0, sigma, size=len(noisy))
                 noisy[col] += noise
        return noisy

    # Create duplicates
    X_high_aug = add_noise(X_high)
    X_low_aug = add_noise(X_low)
    
    # Concat
    X_aug = pd.concat([X, X_high_aug, X_low_aug], axis=0)
    y_aug = pd.concat([y, y_high, y_low], axis=0)
    
    return X_aug, y_aug

def train_v16_augmented():
    print("Loading Data for V16 Augmented Training...")
    train_df, test_df, submission = load_data()
    
    # Use V15 features (bins, interactions)
    train_final, test_final = preprocess_v15_features(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final
    
    # XGBoost Params (Standard V7/V13 params)
    model_params = {
        'n_estimators': 2000,
        'learning_rate': 0.05,
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
    
    print("\nStarting Augmented Training...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # Validation Set (Strictly Keep Clean)
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Training Set (Augment THIS)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        X_train_aug, y_train_aug = augment_data(X_train, y_train)
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(
            X_train_aug, y_train_aug,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_preds = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_preds
        test_preds += np.clip(model.predict(X_test), 0, 100) / 5
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nV16 Overall CV RMSE: {overall_rmse:.4f}")
    
    # Save OOF
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oof_df = train_df[['id']].copy()
    oof_df['exam_score'] = y
    oof_df['pred'] = oof_preds
    oof_path = os.path.join(base_dir, 'submissions', 'oof_v16_aug.csv')
    oof_df.to_csv(oof_path, index=False)
    
    # Save submission
    sub_path = os.path.join(base_dir, 'submissions', 'submission_v16_aug.csv')
    submission['exam_score'] = test_preds
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_v16_augmented()
