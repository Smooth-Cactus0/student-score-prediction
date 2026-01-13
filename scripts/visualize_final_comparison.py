import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_final_comparison():
    # Data manually aggregated from previous runs to ensure accuracy without re-running everything
    data = {
        'Model': ['V13 (Teacher)', 'V16 (Augmented)', 'V21 (Mega Stack)', 'V23 (Hybrid)', 'V24 (Feat Boost)'],
        'CV RMSE': [8.7306, 8.8362, 8.7267, 8.7266, 8.7302],
        'Grade A MAE': [8.52, 7.66, 8.50, 8.48, 8.51],
        'Grade F MAE': [7.05, 6.78, 7.02, 7.01, 7.04]
    }
    
    df = pd.DataFrame(data)
    
    # Melt for plotting
    df_melt = df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Create a custom palette
    palette = {'CV RMSE': 'gray', 'Grade A MAE': '#d62728', 'Grade F MAE': '#1f77b4'}
    
    ax = sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', palette=palette)
    
    # Add labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
        
    plt.title('Final Model Comparison: Overall Accuracy vs Extreme Performance')
    plt.ylabel('Error (Lower is Better)')
    plt.ylim(6, 9) # Zoom in to show differences
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, 'images', 'final_model_comparison.png')
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")

if __name__ == "__main__":
    plot_final_comparison()
