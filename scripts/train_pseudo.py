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

def train_pseudo():
    print("Loading data for Pseudo-Labeling...")
    train_df, test_df, submission = load_data()
    
    # Load the best submission to use as pseudo-labels
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pseudo_source_path = os.path.join(base_dir, 'submissions', 'submission_v7_tuned.csv')
    
    if not os.path.exists(pseudo_source_path):
        print(f"Error: {pseudo_source_path} not found. Run V7 tuning first.")
        return

    print(f"Using pseudo-labels from: {os.path.basename(pseudo_source_path)}")
    pseudo_labels = pd.read_csv(pseudo_source_path)
    
    # Preprocess everything
    train_final, test_final = preprocess_v7(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final
    
    # Create the Pseudo-Training Set (Test features + Predicted labels)
    X_pseudo = X_test.copy()
    y_pseudo = pseudo_labels['exam_score']
    
    print(f"Augmenting training with {len(X_pseudo)} pseudo-labeled samples.")

    # Model Params (Same as V7 Best)
    model_params = {
        'n_estimators': 1500,
        'learning_rate': 0.056,
        'max_depth': 6,
        'subsample': 0.82,
        'colsample_bytree': 0.60,
        'min_child_weight': 14,
        'reg_alpha': 2.5,
        'reg_lambda': 0.03,
        'gamma': 0.1,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        # 1. Real Training Data for this fold
        X_tr_real = X.iloc[train_idx]
        y_tr_real = y.iloc[train_idx]
        
        # 2. Validation Data (Strictly Real Data Only)
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # 3. Combine Real + Pseudo for Training
        X_tr_combined = pd.concat([X_tr_real, X_pseudo], axis=0)
        y_tr_combined = pd.concat([y_tr_real, y_pseudo], axis=0)
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(
            X_tr_combined, y_tr_combined,
            eval_set=[(X_val, y_val)], # Validating on real data prevents confirmation bias metrics
            verbose=False
        )
        
        val_preds = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_preds
        test_preds += np.clip(model.predict(X_test), 0, 100) / 5
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nPseudo-Labeling Overall CV RMSE: {overall_rmse:.4f}")
    
    # Save submission
    sub_path = os.path.join(base_dir, 'submissions', 'submission_v8_pseudo.csv')
    submission['exam_score'] = test_preds
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_pseudo()
