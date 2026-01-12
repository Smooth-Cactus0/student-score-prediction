import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def train_stacking_v21():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    
    # Load OOFs
    oofs = {
        'v13': 'oof_v13_teacher.csv',
        'lgbm': 'oof_lgbm.csv',
        'ann': 'oof_ann.csv',
        'hgb': 'oof_hgb.csv',
        'v16': 'oof_v16_aug.csv',
        'v20': 'oof_v20_meta_teacher.csv'
    }
    
    df_meta = pd.DataFrame()
    y = None
    
    for name, filename in oofs.items():
        path = os.path.join(sub_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df_meta[name] = df['pred']
            if y is None:
                y = df['exam_score']
        else:
            print(f"Warning: {filename} not found.")
            
    # Prepare Test Meta-Features
    subs = {
        'v13': 'submission_v13_teacher.csv',
        'lgbm': 'submission_lgbm.csv',
        'ann': 'submission_ann.csv',
        'hgb': 'submission_hgb.csv',
        'v16': 'submission_v16_aug.csv',
        'v20': 'submission_v20_meta_teacher.csv'
    }
    
    df_test_meta = pd.DataFrame()
    for name, filename in subs.items():
        path = os.path.join(sub_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df_test_meta[name] = df['exam_score']
            
    # Cross-validated Stacking
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_stack = np.zeros(len(y))
    test_stack = np.zeros(len(df_test_meta))
    
    print(f"Training Ridge Stacker (V21) with {len(df_meta.columns)} models...")
    for train_idx, val_idx in kf.split(df_meta, y):
        X_train, X_val = df_meta.iloc[train_idx], df_meta.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        stacker = Ridge(alpha=1.0)
        stacker.fit(X_train, y_train)
        
        oof_stack[val_idx] = stacker.predict(X_val)
        test_stack += stacker.predict(df_test_meta) / 5
        
    rmse = np.sqrt(mean_squared_error(y, oof_stack))
    print(f"V21 Mega Stacking CV RMSE: {rmse:.4f}")
    
    # Save OOF
    oof_df = pd.DataFrame({'id': pd.read_csv(os.path.join(sub_dir, 'oof_v13_teacher.csv'))['id'], 'exam_score': y, 'pred': oof_stack})
    oof_path = os.path.join(sub_dir, 'oof_v21_mega_stacking.csv')
    oof_df.to_csv(oof_path, index=False)
    print(f"V21 OOF saved to '{oof_path}'")
    
    # Coefficients
    final_stacker = Ridge(alpha=1.0)
    final_stacker.fit(df_meta, y)
    print("\nRidge Coefficients:")
    for col, coef in zip(df_meta.columns, final_stacker.coef_):
        print(f"{col}: {coef:.4f}")
    print(f"Intercept: {final_stacker.intercept_:.4f}")
    
    # Save Submission
    sub_sample = pd.read_csv(os.path.join(base_dir, 'data', 'sample_submission.csv'))
    sub_sample['exam_score'] = np.clip(test_stack, 0, 100)
    sub_path = os.path.join(sub_dir, 'submission_v21_mega_stacking.csv')
    sub_sample.to_csv(sub_path, index=False)
    print(f"V21 Mega Stacking submission saved to '{sub_path}'")

if __name__ == "__main__":
    train_stacking_v21()
