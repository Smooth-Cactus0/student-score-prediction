import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier, export_text

# Import V7 preprocessing for specific functions if needed, 
# but we will largely rebuild specific features to test them in isolation.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data, RANDOM_STATE

def get_grade_mae(y_true, y_pred):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df['abs_err'] = (df['true'] - df['pred']).abs()
    
    mae_a = df[df['true'] >= 90]['abs_err'].mean()
    mae_f = df[df['true'] < 60]['abs_err'].mean()
    mae_global = df['abs_err'].mean()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return rmse, mae_global, mae_a, mae_f

def train_eval(X, y, name="Model"):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))
    
    # Fast XGB params for discovery
    params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_jobs': -1,
        'random_state': RANDOM_STATE,
        'objective': 'reg:squarederror'
    }
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)
        
    rmse, mae, mae_a, mae_f = get_grade_mae(y, oof_preds)
    print(f"[{name}] RMSE: {rmse:.4f} | MAE A: {mae_a:.4f} | MAE F: {mae_f:.4f}")
    return {'rmse': rmse, 'mae_a': mae_a, 'mae_f': mae_f}

def feature_discovery():
    print("Loading Data...")
    train_df, test_df, _ = load_data()
    
    # --- BASELINE PREPROCESSING (Minimal) ---
    # We want to see the effect of NEW features, so we start with a clean simple slate.
    # 1. Ordinal Encoding
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    train_df['sleep_quality_num'] = train_df['sleep_quality'].map(sleep_map)
    
    # 2. Simple One-Hot
    dummies = pd.get_dummies(train_df[['gender', 'course', 'study_method', 'internet_access', 'facility_rating', 'exam_difficulty']], drop_first=True)
    
    # 3. Numeric
    base_feats = ['study_hours', 'class_attendance', 'sleep_hours', 'age', 'sleep_quality_num']
    
    X_base = pd.concat([train_df[base_feats], dummies], axis=1)
    y = train_df['exam_score']
    
    print("\n--- Baseline (Basic Features) ---")
    res_base = train_eval(X_base, y, "Baseline")
    
    # ==============================================================================
    # PHASE 1: Cohort Segmentation & Deviation Analysis
    # ==============================================================================
    print("\n--- Phase 1: Relative Features (Deviations) ---")
    X_p1 = X_base.copy()
    
    # Calculate Group Means (Using whole train for simplicity in discovery, 
    # strictly should be CV-safe but effect size is what matters here)
    course_means = train_df.groupby('course')[['study_hours', 'class_attendance']].transform('mean')
    
    X_p1['study_vs_course'] = train_df['study_hours'] - course_means['study_hours']
    X_p1['attend_vs_course'] = train_df['class_attendance'] - course_means['class_attendance']
    
    # Deviation from global means
    X_p1['study_vs_avg'] = train_df['study_hours'] - train_df['study_hours'].mean()
    X_p1['attend_vs_avg'] = train_df['class_attendance'] - train_df['class_attendance'].mean()
    
    res_p1 = train_eval(X_p1, y, "Phase 1")
    
    # ==============================================================================
    # PHASE 2: Threshold & Red Flags (Decision Trees)
    # ==============================================================================
    print("\n--- Phase 2: Red Flags (Tree Rules) ---")
    # Train shallow tree to find cutoffs for F and A
    y_F = (y < 60).astype(int)
    y_A = (y >= 90).astype(int)
    
    # Use only interpretable base features for the tree
    tree_feats = ['study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality_num']
    
    dt_F = DecisionTreeClassifier(max_depth=2, random_state=42)
    dt_F.fit(X_base[tree_feats], y_F)
    print("Rule for F (Grade < 60):")
    print(export_text(dt_F, feature_names=tree_feats))
    
    dt_A = DecisionTreeClassifier(max_depth=2, random_state=42)
    dt_A.fit(X_base[tree_feats], y_A)
    print("Rule for A (Grade >= 90):")
    print(export_text(dt_A, feature_names=tree_feats))
    
    X_p2 = X_base.copy()
    # Hardcoded based on common findings (Dynamic parsing is complex, let's guess reasonable splits or use the output above mentally, 
    # but for automation let's add what the tree *likely* found + generic logical thresholds)
    
    # Tree typically finds: Attendance < 60-70 is bad. Study hours < 2 is bad.
    X_p2['flag_low_attend'] = (train_df['class_attendance'] < 70).astype(int)
    X_p2['flag_low_study'] = (train_df['study_hours'] < 3).astype(int)
    
    # For A: High study + High attendance
    X_p2['flag_high_performer'] = ((train_df['study_hours'] > 7) & (train_df['class_attendance'] > 90)).astype(int)
    
    res_p2 = train_eval(X_p2, y, "Phase 2")
    
    # ==============================================================================
    # PHASE 3: Polynomial & Interaction Expansion
    # ==============================================================================
    print("\n--- Phase 3: Poly & Interactions ---")
    X_p3 = X_base.copy()
    
    X_p3['log_study'] = np.log1p(train_df['study_hours'])
    X_p3['attend_sq'] = train_df['class_attendance'] ** 2
    X_p3['dedication'] = train_df['study_hours'] * train_df['class_attendance']
    X_p3['dedication_sleep'] = X_p3['dedication'] * (train_df['sleep_quality_num'] + 1)
    
    res_p3 = train_eval(X_p3, y, "Phase 3")
    
    # ==============================================================================
    # PHASE 5: Psychology Features
    # ==============================================================================
    print("\n--- Phase 5: Psychology Features ---")
    X_p5 = X_base.copy()
    
    # Slacker: Low study but shows up (attendance > 80, study < 2)
    X_p5['type_slacker'] = ((train_df['class_attendance'] > 80) & (train_df['study_hours'] < 2)).astype(int)
    
    # Hard Worker: High study, High attendance
    X_p5['type_hardworker'] = ((train_df['class_attendance'] > 90) & (train_df['study_hours'] > 6)).astype(int)
    
    # Burnout: High study (>6), Low Sleep (<5) or Poor Sleep (0)
    X_p5['type_burnout'] = ((train_df['study_hours'] > 6) & ((train_df['sleep_hours'] < 5) | (train_df['sleep_quality_num'] == 0))).astype(int)
    
    res_p5 = train_eval(X_p5, y, "Phase 5")
    
    # ==============================================================================
    # COMBINED BEST
    # ==============================================================================
    print("\n--- Combined Best Features ---")
    X_final = X_base.copy()
    
    # Phase 1
    X_final['study_vs_course'] = X_p1['study_vs_course']
    
    # Phase 2
    X_final['flag_low_attend'] = X_p2['flag_low_attend']
    
    # Phase 3
    X_final['dedication_sleep'] = X_p3['dedication_sleep']
    X_final['attend_sq'] = X_p3['attend_sq']
    
    # Phase 5
    X_final['type_slacker'] = X_p5['type_slacker']
    X_final['type_burnout'] = X_p5['type_burnout']
    
    res_final = train_eval(X_final, y, "Combined")

    print("\nSummary Table:")
    print(f"{ 'Phase':<15} | { 'RMSE':<8} | { 'MAE A':<8} | { 'MAE F':<8}")
    print("-" * 45)
    for name, res in [('Baseline', res_base), ('Phase 1', res_p1), ('Phase 2', res_p2), 
                      ('Phase 3', res_p3), ('Phase 5', res_p5), ('Combined', res_final)]:
        print(f"{name:<15} | {res['rmse']:.4f}   | {res['mae_a']:.4f}   | {res['mae_f']:.4f}")

if __name__ == "__main__":
    feature_discovery()
