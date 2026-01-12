import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def train_stacking_v19():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    
    # Load OOFs
    oof_v13 = pd.read_csv(os.path.join(sub_dir, 'oof_v13_teacher.csv'))
    oof_lgbm = pd.read_csv(os.path.join(sub_dir, 'oof_lgbm.csv'))
    oof_ann = pd.read_csv(os.path.join(sub_dir, 'oof_ann.csv'))
    oof_v16 = pd.read_csv(os.path.join(sub_dir, 'oof_v16_aug.csv'))
    
    y = oof_v13['exam_score']
    
    # Prepare Meta-Features
    X_meta = pd.DataFrame({
        'v13': oof_v13['pred'],
        'lgbm': oof_lgbm['pred'],
        'ann': oof_ann['pred'],
        'v16': oof_v16['pred']
    })
    
    # Load Submissions
    sub_v13 = pd.read_csv(os.path.join(sub_dir, 'submission_v13_teacher.csv'))
    sub_lgbm = pd.read_csv(os.path.join(sub_dir, 'submission_lgbm.csv'))
    sub_ann = pd.read_csv(os.path.join(sub_dir, 'submission_ann.csv'))
    sub_v16 = pd.read_csv(os.path.join(sub_dir, 'submission_v16_aug.csv'))
    
    X_test_meta = pd.DataFrame({
        'v13': sub_v13['exam_score'],
        'lgbm': sub_lgbm['exam_score'],
        'ann': sub_ann['exam_score'],
        'v16': sub_v16['exam_score']
    })
    
    # Cross-validated Stacking
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_stack = np.zeros(len(y))
    test_stack = np.zeros(len(X_test_meta))
    
    print("Training Ridge Stacker (V19)...")
    for train_idx, val_idx in kf.split(X_meta, y):
        X_train, X_val = X_meta.iloc[train_idx], X_meta.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        stacker = Ridge(alpha=1.0)
        stacker.fit(X_train, y_train)
        
        oof_stack[val_idx] = stacker.predict(X_val)
        test_stack += stacker.predict(X_test_meta) / 5
        
    rmse = np.sqrt(mean_squared_error(y, oof_stack))
    print(f"V19 Stacking CV RMSE: {rmse:.4f}")
    
    # Coefficients
    final_stacker = Ridge(alpha=1.0)
    final_stacker.fit(X_meta, y)
    print("\nRidge Coefficients:")
    for col, coef in zip(X_meta.columns, final_stacker.coef_):
        print(f"{col}: {coef:.4f}")
    print(f"Intercept: {final_stacker.intercept_:.4f}")
    
    # Save Submission
    submission = sub_v13.copy()
    submission['exam_score'] = np.clip(test_stack, 0, 100)
    sub_path = os.path.join(sub_dir, 'submission_v19_stacking.csv')
    submission.to_csv(sub_path, index=False)
    print(f"V19 Stacking submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_stacking_v19()
