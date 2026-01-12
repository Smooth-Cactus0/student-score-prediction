import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

def plot_stack_correlation():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    img_dir = os.path.join(base_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    # OOF files used in v21
    oofs = {
        'V13 (Teacher)': 'oof_v13_teacher.csv',
        'LGBM': 'oof_lgbm.csv',
        'ANN': 'oof_ann.csv',
        'HGB': 'oof_hgb.csv',
        'V16 (Augmented)': 'oof_v16_aug.csv',
        'V20 (Meta)': 'oof_v20_meta_teacher.csv',
        'V21 (Mega Stack)': 'oof_v21_mega_stacking.csv'
    }
    
    df_corr = pd.DataFrame()
    
    print("Loading OOFs for correlation analysis...")
    for name, filename in oofs.items():
        path = os.path.join(sub_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df_corr[name] = df['pred']
        else:
            print(f"Skipping {name} (File not found)")
            
    if df_corr.empty:
        print("No data found.")
        return

    # Compute Correlation
    corr_matrix = df_corr.corr()
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".4f", vmin=0.9, vmax=1.0)
    plt.title('Correlation Matrix of Stacking Components (V21)')
    plt.tight_layout()
    
    save_path = os.path.join(img_dir, 'stack_correlation_v21.png')
    plt.savefig(save_path)
    print(f"Correlation plot saved to '{save_path}'")
    
    print("\nPairwise Correlations with V21 (Final):")
    print(corr_matrix['V21 (Mega Stack)'].sort_values(ascending=False))

if __name__ == "__main__":
    plot_stack_correlation()
