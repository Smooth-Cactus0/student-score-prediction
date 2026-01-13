import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def get_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'

def analyze_failures():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    data_dir = os.path.join(base_dir, 'data')
    img_dir = os.path.join(base_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    # Load Train Data for Feature Context
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    # Key Models
    models = {
        'V13 (XGB)': 'oof_v13_teacher.csv',
        'LGBM': 'oof_lgbm.csv',
        'ANN': 'oof_ann.csv',
        'V16 (Aug)': 'oof_v16_aug.csv',
        'V21 (Stack)': 'oof_v21_mega_stacking.csv'
    }
    
    df_res = pd.DataFrame()
    df_preds = pd.DataFrame()
    
    print("Loading OOFs...")
    first = True
    for name, filename in models.items():
        path = os.path.join(sub_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if first:
                df_res['Actual'] = df['exam_score']
                df_res['Grade'] = df['exam_score'].apply(get_grade)
                df_res['id'] = df['id'] # Assuming ID exists or we use index
                first = False
            
            df_preds[name] = df['pred']
            # Residual = Actual - Predicted
            # Positive = Underprediction (Score was 90, pred 80 -> +10)
            # Negative = Overprediction (Score was 50, pred 60 -> -10)
            df_res[f'Res_{name}'] = df['exam_score'] - df['pred']
        else:
            print(f"Warning: {filename} not found")

    # 1. Bias Analysis (Mean Residual by Grade)
    print("\n--- Bias Analysis (Mean Residual: Actual - Pred) ---")
    print("Positive = Underprediction (Model too pessimistic)")
    print("Negative = Overprediction (Model too optimistic)")
    
    bias_df = df_res.groupby('Grade')[[f'Res_{name}' for name in models]].mean()
    # Sort index A->F
    bias_df = bias_df.reindex(['A', 'B', 'C', 'D', 'F'])
    print(bias_df.round(2))
    
    # 2. Residual Correlation
    print("\n--- Residual Correlation (Do they make the same errors?)")
    res_corr = df_res[[f'Res_{name}' for name in models]].corr()
    print(res_corr.round(4))
    
    # 3. Hardest Samples (Systematic Failure)
    # Calculate Mean Absolute Error across all models for each row
    mae_cols = []
    for name in models:
        df_res[f'Abs_{name}'] = df_res[f'Res_{name}'].abs()
        mae_cols.append(f'Abs_{name}')
        
    df_res['Mean_Model_MAE'] = df_res[mae_cols].mean(axis=1)
    
    # Top 20 Hardest Rows
    hardest = df_res.sort_values('Mean_Model_MAE', ascending=False).head(20)
    
    # Join with features
    hardest_features = hardest.merge(train_df, left_on='id', right_on='id', how='left')
    
    print("\n--- Top 10 Hardest Samples to Predict (Systematic Failure) ---")
    cols_to_show = ['id', 'exam_score', 'Mean_Model_MAE', 'study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality'] 
    # Add model preds for context
    for name in models:
        cols_to_show.append(f'Res_{name}')
        
    print(hardest_features[cols_to_show].head(10).to_string(index=False))
    
    # 4. Feature Analysis of Errors
    # Merge errors back to full train df to see correlations
    full_analysis = train_df.merge(df_res[['id', 'Mean_Model_MAE', 'Res_V13 (XGB)']], on='id')
    
    print("\n--- Correlation of Features with Absolute Error (V13) ---")
    # Select numeric columns only
    numeric_cols = full_analysis.select_dtypes(include=[np.number]).columns
    corr_with_error = full_analysis[numeric_cols].corr()['Mean_Model_MAE'].sort_values(ascending=False)
    print(corr_with_error.head(5))
    print(corr_with_error.tail(5))

    # Visualization: Bias Plot
    plt.figure(figsize=(10, 6))
    bias_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Model Bias by Grade (Actual - Pred)')
    plt.ylabel('Mean Residual')
    plt.axhline(0, color='k', linewidth=0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'model_bias_analysis.png'))
    print(f"\nBias plot saved to {os.path.join(img_dir, 'model_bias_analysis.png')}")

if __name__ == "__main__":
    analyze_failures()
