import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_data, load_data, RANDOM_STATE

def objective(trial):
    # Load data only once globally if possible, but inside objective is safer for multiprocessing 
    # (though Optuna uses threads/processes differently).
    # For simplicity, we assume data is fast to load or we load it outside.
    pass

# Global load for speed
train_df, test_df, submission = load_data()
train_final, test_final = preprocess_data(train_df, test_df)
X = train_final.drop(columns=['exam_score'])
y = train_final['exam_score']
X_test = test_final

def tune_xgboost():
    print("Starting Optuna Tuning...")
    
    def objective(trial):
        params = {
            'n_estimators': 1000, # Fixed or small range
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'n_jobs': -1,
            'objective': 'reg:squarederror',
            'random_state': RANDOM_STATE,
            'early_stopping_rounds': 50
        }
        
        # Fast CV (3-Fold) for tuning
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        rmse_scores = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            rmse_scores.append(rmse)
        
        return np.mean(rmse_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15) # Small number for demo speed
    
    print("\nBest Params:")
    print(study.best_params)
    
    # Train Final Model with Best Params (on full 5-fold)
    print("\nTraining Final Model (5-Fold)...")
    best_params = study.best_params
    best_params['n_estimators'] = 1500 # Increase for final run
    best_params['n_jobs'] = -1
    best_params['objective'] = 'reg:squarederror'
    best_params['random_state'] = RANDOM_STATE
    best_params['early_stopping_rounds'] = 50
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    test_preds = np.zeros(len(X_test))
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # OOF Score
        val_preds = np.clip(model.predict(X_val), 0, 100)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        cv_scores.append(rmse)
        
        test_preds += np.clip(model.predict(X_test), 0, 100) / 5
        print(f"Final Fold {fold+1} RMSE: {rmse:.4f}")

    print(f"\nFinal Optuna CV RMSE: {np.mean(cv_scores):.4f}")
    
    submission['exam_score'] = test_preds
    
    # Robust path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    os.makedirs(sub_dir, exist_ok=True)
    
    save_path = os.path.join(sub_dir, 'submission_optuna_v5.csv')
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to '{save_path}'")

if __name__ == "__main__":
    tune_xgboost()
