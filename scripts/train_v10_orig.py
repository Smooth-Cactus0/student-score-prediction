import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Import V7 preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_v7 import preprocess_v7
from preprocessing import load_data, RANDOM_STATE

def train_with_original():
    print("Loading Competition Data...")
    train_df, test_df, submission = load_data()
    
    print("Loading Original Dataset...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orig_path = os.path.join(base_dir, 'archive', 'Exam_Score_Prediction.csv')
    
    if not os.path.exists(orig_path):
        print(f"Error: Original data not found at {orig_path}")
        return

    orig_df = pd.read_csv(orig_path)
    
    # Align columns
    # Original has 'student_id', competition has 'id'
    orig_df.rename(columns={'student_id': 'id'}, inplace=True)
    
    # Add source indicator (optional, but good practice if distributions vary)
    train_df['is_original'] = 0
    orig_df['is_original'] = 1
    test_df['is_original'] = 0
    
    # Concatenate
    print(f"Competition Train Shape: {train_df.shape}")
    print(f"Original Data Shape: {orig_df.shape}")
    
    combined_train = pd.concat([train_df, orig_df], axis=0).reset_index(drop=True)
    print(f"Combined Train Shape: {combined_train.shape}")
    
    # Preprocess
    # We need to pass the combined train to preprocess_v7
    # preprocess_v7 expects train and test. It combines them internally.
    # It drops 'id' and 'is_train' (if it exists, logic inside creates it).
    # The 'is_original' column will be treated as a feature if we don't drop it or if preprocess_v7 passes it through.
    # preprocess_v7 drops specific columns: ['id', 'sleep_quality', 'facility_rating', 'exam_difficulty'] and returns numeric.
    # It does pd.get_dummies. 'is_original' is numeric (0/1), so it will persist.
    
    train_final, test_final = preprocess_v7(combined_train, test_df)
    
    # Check if 'is_original' survived (it should have)
    if 'is_original' in train_final.columns:
        print("Feature 'is_original' included.")
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final
    
    # Model Params (V7 Tuned / Conservative)
    model_params = {
        'n_estimators': 2000,
        'learning_rate': 0.04,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        'reg_alpha': 2.0,
        'reg_lambda': 0.1,
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
        
        # We generally want to validate mainly on the COMPETITION distribution
        # But for standard CV, we just mix. 
        # Advanced: Validate only on is_original==0 samples to match leaderboard distribution?
        # Let's try standard CV first. If the original data is "real", it's good.
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_preds = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_preds
        test_preds += np.clip(model.predict(X_test), 0, 100) / 5
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nCombined Data Overall CV RMSE: {overall_rmse:.4f}")
    
    # Calculate RMSE on Competition Data Only (True Benchmark)
    # We need to access the 'is_original' column from X to filter
    if 'is_original' in X.columns:
        comp_mask = X['is_original'] == 0
        comp_y = y[comp_mask]
        comp_preds = oof_preds[comp_mask]
        comp_rmse = np.sqrt(mean_squared_error(comp_y, comp_preds))
        print(f"Competition Data Only CV RMSE: {comp_rmse:.4f} (Benchmark vs V7)")
    
    print("\n--- Feature Importance ---")
    importances = pd.Series(0.0, index=X.columns)
    # We didn't store model per fold, so we can't avg perfectly, 
    # but we can grab the last model's importance as a proxy or retrain.
    # Actually, simpler: just use the last model trained in the loop.
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print(importances.sort_values(ascending=False).head(10))

    # Save submission
    sub_path = os.path.join(base_dir, 'submissions', 'submission_v10_orig.csv')
    submission['exam_score'] = test_preds
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_with_original()
