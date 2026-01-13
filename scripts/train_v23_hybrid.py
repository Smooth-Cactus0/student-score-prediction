import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score

# Import V7 preprocessing for the Classifier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_v7 import preprocess_v7
from preprocessing import load_data, RANDOM_STATE

def train_v23_hybrid():
    print("Loading Data...")
    train_df, test_df, sub_sample = load_data()
    
    # 1. Preprocess for Classifier (Using V7 features as they are robust)
    train_final, test_final = preprocess_v7(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y_score = train_final['exam_score']
    
    # Define Target for Classifier: Extreme (Score < 60 OR Score > 85)
    # Adjusting thresholds slightly to capture the "problematic" zones
    y_class = ((y_score < 60) | (y_score > 85)).astype(int)
    
    # 2. Train Classifier to get P(Extreme)
    clf_params = {
        'n_estimators': 300,
        'learning_rate': 0.04,
        'max_depth': 5,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_prob_extreme = np.zeros(len(X))
    test_prob_extreme = np.zeros(len(test_final))
    
    print(f"Training Classifier (Extreme Ratio: {y_class.mean():.2%})...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_class)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_class.iloc[train_idx], y_class.iloc[val_idx]
        
        clf = xgb.XGBClassifier(**clf_params)
        clf.fit(X_train, y_train, verbose=False)
        
        oof_prob_extreme[val_idx] = clf.predict_proba(X_val)[:, 1]
        test_prob_extreme += clf.predict_proba(test_final)[:, 1] / 5
        
        auc = roc_auc_score(y_val, oof_prob_extreme[val_idx])
        # print(f"Fold {fold+1} AUC: {auc:.4f}")

    print(f"Overall AUC: {roc_auc_score(y_class, oof_prob_extreme):.4f}")
    
    # 3. Load Regressor OOFs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    
    # Safe Model (V21 Mega Stack)
    oof_v21 = pd.read_csv(os.path.join(sub_dir, 'oof_v21_mega_stacking.csv'))
    sub_v21 = pd.read_csv(os.path.join(sub_dir, 'submission_v21_mega_stacking.csv'))
    
    # Risky/Augmented Model (V16 or V20) - V16 was better at extremes
    oof_v16 = pd.read_csv(os.path.join(sub_dir, 'oof_v16_aug.csv'))
    sub_v16 = pd.read_csv(os.path.join(sub_dir, 'submission_v16_aug.csv'))
    
    # Ensure alignment
    # (Assuming id order is preserved, which it is for standard KFold on same data)
    
    # 4. Hybrid Blending
    # Formula: Final = (1 - P) * Safe + P * Risky
    # But P might need scaling. If P=0.6, do we want 60% risky? Maybe.
    
    # Let's optimize the mixing weight 'w' such that:
    # Final = (1 - w*P) * Safe + (w*P) * Risky
    # where P is the probability of being extreme.
    
    best_rmse = 999
    best_w = 0
    
    y_true = oof_v21['exam_score']
    pred_safe = oof_v21['pred']
    pred_risky = oof_v16['pred']
    
    # Grid search for weight scalar
    print("\nOptimizing Hybrid Weight...")
    for w in np.linspace(0.0, 1.5, 31):
        # Weight for risky model depends on confidence it's extreme
        weight_risky = np.clip(w * oof_prob_extreme, 0, 1)
        
        hybrid_pred = (1 - weight_risky) * pred_safe + weight_risky * pred_risky
        rmse = np.sqrt(mean_squared_error(y_true, hybrid_pred))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w
            
    print(f"Best Weight Scalar (w): {best_w:.2f}")
    print(f"Best Hybrid CV RMSE: {best_rmse:.4f} (V21 was 8.7267)")
    
    # 5. Generate Final Predictions
    final_weight_risky = np.clip(best_w * test_prob_extreme, 0, 1)
    final_test_preds = (1 - final_weight_risky) * sub_v21['exam_score'] + final_weight_risky * sub_v16['exam_score']
    
    # 6. Analysis of Bias
    oof_weight_risky = np.clip(best_w * oof_prob_extreme, 0, 1)
    oof_hybrid = (1 - oof_weight_risky) * pred_safe + oof_weight_risky * pred_risky
    
    df_res = pd.DataFrame({'exam_score': y_true, 'pred': oof_hybrid})
    df_res['AbsError'] = (df_res['exam_score'] - df_res['pred']).abs()
    
    def get_grade(s):
        if s >= 90: return 'A'
        elif s < 60: return 'F'
        else: return 'Other'
        
    df_res['Grade'] = df_res['exam_score'].apply(get_grade)
    
    print("\nHybrid Bias Analysis (MAE):")
    print(df_res.groupby('Grade')['AbsError'].mean())
    
    # Save
    sub_path = os.path.join(sub_dir, 'submission_v23_hybrid.csv')
    sub_sample['exam_score'] = np.clip(final_test_preds, 0, 100)
    sub_sample.to_csv(sub_path, index=False)
    print(f"Submission saved to '{sub_path}'")
    
    # Save OOF for future stacking
    oof_save = oof_v21[['id']].copy()
    oof_save['exam_score'] = y_true
    oof_save['pred'] = oof_hybrid
    oof_save.to_csv(os.path.join(sub_dir, 'oof_v23_hybrid.csv'), index=False)

if __name__ == "__main__":
    train_v23_hybrid()
