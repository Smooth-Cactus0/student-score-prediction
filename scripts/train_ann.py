import pandas as pd
import numpy as np
import os
import sys
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess_data, load_data, RANDOM_STATE

def train_ann():
    print("Loading and Preprocessing...")
    train_df, test_df, submission = load_data()
    
    train_final, test_final = preprocess_data(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y = train_final['exam_score']
    X_test = test_final
    
    # --- Scaling (Critical for ANN) ---
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # --- Modeling (MLP) ---
    print("\n--- Training ANN (MLPRegressor) ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    rmse_scores = []
    
    mlp_params = {
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 128,
        'learning_rate_init': 0.001,
        'max_iter': 200,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'random_state': RANDOM_STATE
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = MLPRegressor(**mlp_params)
        model.fit(X_train, y_train)
        
        val_preds = np.clip(model.predict(X_val), 0, 100)
        oof_preds[val_idx] = val_preds
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        rmse_scores.append(rmse)
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")
        
        test_preds += np.clip(model.predict(X_test_scaled), 0, 100) / 5

    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nANN Overall CV RMSE: {overall_rmse:.4f}")
    
    # --- Save ANN Submission ---
    submission['exam_score'] = test_preds
    
    # Robust path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    os.makedirs(sub_dir, exist_ok=True)
    
    save_path = os.path.join(sub_dir, 'submission_ann.csv')
    submission.to_csv(save_path, index=False)
    print(f"ANN Submission saved to '{save_path}'")

if __name__ == "__main__":
    train_ann()
