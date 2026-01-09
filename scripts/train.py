import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def preprocess_data(df, is_train=True):
    # --- 1. Ordinal Encoding ---
    # Map qualitative variables to numbers based on logical order
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_map = {'low': 0, 'medium': 1, 'high': 2}
    difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    
    df['sleep_quality_num'] = df['sleep_quality'].map(sleep_map)
    df['facility_rating_num'] = df['facility_rating'].map(facility_map)
    df['exam_difficulty_num'] = df['exam_difficulty'].map(difficulty_map)
    
    # --- 2. Feature Interactions (The "Dedication" features) ---
    # Hypothesis: Study hours matter more if you also attend class
    df['study_attendance_interaction'] = df['study_hours'] * df['class_attendance']
    
    # Hypothesis: Sleep efficiency - study hours per unit of sleep
    # avoiding division by zero with clip or small epsilon
    df['study_per_sleep'] = df['study_hours'] / (df['sleep_hours'] + 0.1)
    
    return df

def train_and_evaluate():
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    submission = pd.read_csv('sample_submission.csv')

    print("Preprocessing...")
    # Combine for consistent categorical encoding if needed, 
    # but for OneHot/Target encoding we need to be careful.
    # Here we'll just process independently for the deterministic stuff.
    train = preprocess_data(train, is_train=True)
    test = preprocess_data(test, is_train=False)
    
    # --- 3. Categorical Encoding ---
    # For tree models, simple Label Encoding is often sufficient and efficient.
    # For high cardinality, Target Encoding is better, but 'course' has few levels.
    categorical_cols = ['gender', 'course', 'internet_access', 'study_method']
    
    # We will use One-Hot Encoding for low cardinality nominals to give the model max flexibility
    # Combine to ensure same columns
    all_data = pd.concat([train.drop('exam_score', axis=1), test], axis=0)
    all_data = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)
    
    # Split back
    X = all_data.iloc[:len(train)]
    X_test = all_data.iloc[len(train):]
    y = train['exam_score']
    
    # Drop original text columns that were ordinally encoded
    drop_cols = ['id', 'sleep_quality', 'facility_rating', 'exam_difficulty']
    X = X.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)
    
    print(f"Features used ({len(X.columns)}): {list(X.columns)}")

    # --- 4. Modeling (XGBoost) ---
    print("\nStarting Training (5-Fold CV)...")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    rmse_scores = []
    
    model_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'early_stopping_rounds': 50
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**model_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        
        # Clip predictions to valid range (0-100)
        val_preds = np.clip(val_preds, 0, 100)
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        rmse_scores.append(rmse)
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")
        
        test_preds += model.predict(X_test) / 5

    # Overall Metrics
    overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\nOverall CV RMSE: {overall_rmse:.4f}")
    print(f"Average Fold RMSE: {np.mean(rmse_scores):.4f}")

    # --- Feature Importance Analysis ---
    print("\n--- Feature Importance (Top 20) ---")
    # Get importance from the last trained model (or average if we stored them)
    # Since we didn't store them, we'll just take the last model's importance 
    # (which is a decent proxy, or we could refactor to store all)
    # Better: let's refactor slightly to store importances
    importances = pd.DataFrame(index=X.columns)
    importances['total_gain'] = 0
    
    # We need to rerun or just use the last model? 
    # To be accurate without retraining, we should have stored it. 
    # But for this 'replace' block, I'll just use the last 'model' object from the loop 
    # which is still in memory if we run this script linearly.
    # However, to be robust, let's just print the importance of the last fold's model.
    
    # improved: let's use the built-in plot_importance equivalent logic
    fi = pd.Series(model.feature_importances_, index=X.columns)
    print(fi.sort_values(ascending=False).head(20))

    # --- 5. Submission ---
    submission['exam_score'] = np.clip(test_preds, 0, 100)
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file saved to 'submission.csv'")

if __name__ == "__main__":
    train_and_evaluate()
