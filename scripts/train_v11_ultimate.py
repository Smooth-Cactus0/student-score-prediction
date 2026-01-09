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

def train_ultimate():
    print("Loading all datasets for Ultimate Run...")
    train_df, test_df, submission = load_data()
    
    # 1. Load Original
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orig_path = os.path.join(base_dir, 'archive', 'Exam_Score_Prediction.csv')
    orig_df = pd.read_csv(orig_path)
    orig_df.rename(columns={'student_id': 'id'}, inplace=True)
    
    # 2. Load Pseudo Labels (Source: V7 Tuned)
    pseudo_path = os.path.join(base_dir, 'submissions', 'submission_v7_tuned.csv')
    if not os.path.exists(pseudo_path):
        print("V7 Tuned submission not found.")
        return
    pseudo_df = pd.read_csv(pseudo_path)
    
    print(f"Original: {len(orig_df)}, Pseudo: {len(pseudo_df)}")
    
    # 3. Add Indicators BEFORE concat to track them
    train_df['source'] = 'train'
    orig_df['source'] = 'original'
    test_df['source'] = 'test'
    
    # 4. Global Concat for Preprocessing
    # We need to preprocess everything together to ensure encodings match
    combined_all = pd.concat([train_df, orig_df, test_df], axis=0).reset_index(drop=True)
    
    # Mock 'is_train' for the preprocessor
    # We want preprocessor to treat Train+Original as "Train" (fit encoders) and Test as "Test"
    # But wait, preprocessor splits by 'is_train'.
    # If we pass everything as one, we need to handle it.
    
    # Let's trust preprocess_v7's logic: it takes 'train' and 'test' args.
    # It concats them.
    # So let's construct the inputs it expects.
    
    # Input 1: Train + Original
    train_input = pd.concat([train_df, orig_df], axis=0).reset_index(drop=True)
    
    # Input 2: Test
    test_input = test_df.copy()
    
    print("Preprocessing...")
    train_proc, test_proc = preprocess_v7(train_input, test_input)
    
    # 5. Recover the sources
    # train_proc now contains both Train and Original.
    # We need to distinguish them for Cross-Validation (Validate ONLY on 'train')
    # But 'preprocess_v7' drops the 'source' column if it's not in the allow list.
    # And it does get_dummies.
    
    # Check if 'source' survived. It likely didn't unless we modify v7.
    # Alternatively, we can split by index.
    # train_input was: [Train (630k)] + [Original (20k)]
    n_train = len(train_df)
    n_orig = len(orig_df)
    
    X_train_only = train_proc.iloc[:n_train]
    y_train_only = train_input['exam_score'].iloc[:n_train] # Re-fetch Y carefully
    # Wait, train_proc has 'exam_score' attached at the end of preprocess_v7
    y_train_only = X_train_only['exam_score']
    X_train_only = X_train_only.drop(columns=['exam_score'])
    
    X_orig = train_proc.iloc[n_train:]
    y_orig = X_orig['exam_score']
    X_orig = X_orig.drop(columns=['exam_score'])
    
    X_test_pseudo = test_proc.copy()
    y_test_pseudo = pseudo_df['exam_score']

    # DROP 'source' column (helper)
    if 'source' in X_train_only.columns:
        X_train_only.drop(columns=['source'], inplace=True)
        X_orig.drop(columns=['source'], inplace=True)
        X_test_pseudo.drop(columns=['source'], inplace=True)
    
    print(f"Split Verified: Train={len(X_train_only)}, Orig={len(X_orig)}, Pseudo={len(X_test_pseudo)}")
    
    # 6. Training Loop
    # Model Params (V7 Tuned)
    model_params = {
        'n_estimators': 2000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.82,
        'colsample_bytree': 0.6,
        'min_child_weight': 14,
        'reg_alpha': 2.5,
        'reg_lambda': 0.03,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    test_preds = np.zeros(len(test_proc))
    oof_preds = np.zeros(len(X_train_only))
    
    print("Starting Ultimate Training...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_only, y_train_only)):
        # A. Validation Set (Strictly Competition Train)
        X_val = X_train_only.iloc[val_idx]
        y_val = y_train_only.iloc[val_idx]
        
        # B. Training Set (Competition Train Fold + Original + Pseudo)
        X_tr_comp = X_train_only.iloc[train_idx]
        y_tr_comp = y_train_only.iloc[train_idx]
        
        X_train_full = pd.concat([X_tr_comp, X_orig, X_test_pseudo], axis=0)
        y_train_full = pd.concat([y_tr_comp, y_orig, y_test_pseudo], axis=0)
        
        # Train
        model = xgb.XGBRegressor(**model_params)
        model.fit(
            X_train_full, y_train_full,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict
        val_pred = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_pred
        test_preds += np.clip(model.predict(X_test_pseudo), 0, 100) / 5
        
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y_train_only, oof_preds))
    print(f"\nUltimate (Orig+Pseudo) CV RMSE: {overall_rmse:.4f}")
    
    submission['exam_score'] = test_preds
    sub_path = os.path.join(base_dir, 'submissions', 'submission_v11_ultimate.csv')
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_ultimate()
