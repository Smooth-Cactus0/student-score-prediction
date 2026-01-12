import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_v15_weighted import preprocess_v15_features
from train_v16_aug import augment_data
from preprocessing import load_data, RANDOM_STATE

def objective(trial):
    # Load Data (Cached if possible, but for simplicity reload)
    train_df, test_df, _ = load_data()
    train_final, _ = preprocess_v15_features(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    
    # K-Fold (Use smaller fold count for speed during tuning, e.g., 3)
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    # Hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    }
    
    rmses = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        # Augment Training Data ONLY
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_tr_aug, y_tr_aug = augment_data(X_tr, y_tr)
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr_aug, y_tr_aug,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds = np.clip(model.predict(X_val), 0, 100)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)
    
    return np.mean(rmses)

def run_tuning():
    print("Starting Optuna Tuning for V16 (Augmented).")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20) # 20 trials for demonstration/time
    
    print("\nBest params:")
    print(study.best_params)
    
    # Save best params
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base_dir, 'scripts', 'best_params_v16.txt'), 'w') as f:
        f.write(str(study.best_params))

if __name__ == "__main__":
    run_tuning()
