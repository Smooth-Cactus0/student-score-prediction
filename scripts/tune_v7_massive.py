import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Import our robust V7 preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_v7 import preprocess_v7
from preprocessing import load_data, RANDOM_STATE

def tune_v7():
    print("Loading data for Massive Tuning (V7)...")
    train_df, test_df, submission = load_data()
    train_final, test_final = preprocess_v7(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final

    def objective(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-3, 5.0, log=True),
            'n_jobs': -1,
            'objective': 'reg:squarederror',
            'random_state': RANDOM_STATE,
            'early_stopping_rounds': 50
        }
        
        # 3-Fold CV for speed during 100 trials
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        
        for train_idx, val_idx in kf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            preds = np.clip(model.predict(X_val), 0, 100)
            scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        
        return np.mean(scores)

    print("Starting Optuna Study (30 trials)...")
    study = optuna.create_study(direction='minimize')
    # Reducing trials to fit within the terminal's silent timeout constraints
    study.optimize(objective, n_trials=30) 

    print("\nBest Hyperparameters:")
    print(study.best_params)

    # Final 5-Fold Training
    print("\nTraining Final V7-Tuned Model (5-Fold)...")
    final_params = study.best_params
    final_params.update({
        'n_estimators': 1500, # Balanced for final
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    })

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**final_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100) # Increased verbosity
        
        oof_preds[val_idx] = np.clip(model.predict(X_val), 0, 100)
        test_preds += np.clip(model.predict(X_test), 0, 100) / 5
        print(f"Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y_val, oof_preds[val_idx])):.4f}")

    print(f"\nFinal V7-Tuned CV RMSE: {np.sqrt(mean_squared_error(y, oof_preds)):.4f}")
    
    submission['exam_score'] = test_preds
    sub_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'submissions', 'submission_v7_tuned.csv')
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to '{sub_path}'")

if __name__ == "__main__":
    tune_v7()
