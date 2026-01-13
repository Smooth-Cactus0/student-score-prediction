import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# Import V7 preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_v7 import preprocess_v7
from preprocessing import load_data, RANDOM_STATE

def check_extreme_classifier():
    print("Loading Data...")
    train_df, test_df, _ = load_data()
    train_final, _ = preprocess_v7(train_df, test_df)
    
    X = train_final.drop(columns=['exam_score'])
    y_score = train_final['exam_score']
    
    # Define "Extreme" vs "Normal"
    # Let's say Extreme is < 55 or > 85
    y_class = ((y_score < 55) | (y_score > 85)).astype(int)
    
    print(f"Class Balance: {y_class.mean():.2%} are Extreme")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=RANDOM_STATE
    )
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    
    print("Training Classifier...")
    for train_idx, val_idx in kf.split(X, y_class):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_class.iloc[train_idx], y_class.iloc[val_idx]
        
        model.fit(X_train, y_train, verbose=False)
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"Fold AUC: {auc:.4f}")
        
    overall_auc = roc_auc_score(y_class, oof_preds)
    print(f"\nOverall AUC: {overall_auc:.4f}")
    
    # Feature Importance
    imps = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    print(imps.sort_values('Importance', ascending=False).head(10))

if __name__ == "__main__":
    check_extreme_classifier()
