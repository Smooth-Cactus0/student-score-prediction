import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# Add path to load preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_data, preprocess_data

def analyze_extremes():
    print("Loading data...")
    train_df, _, _ = load_data()
    
    # Create Grade Groups
    train_df['Grade_Group'] = pd.cut(
        train_df['exam_score'], 
        bins=[-1, 59.99, 89.99, 101],
        labels=['F', 'Mid', 'A']
    )
    
    # 1. Compare A vs Rest, F vs Rest
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'exam_score']
    
    print("\n--- Correlations with Grade Groups ---")
    # We'll check the mean values of features for each group
    group_means = train_df.groupby('Grade_Group', observed=True)[numeric_cols].mean()
    print(group_means.T)
    
    # 2. Check for "Super Features"
    # Is there a combination that perfectly separates them?
    
    # Create "Good Habits" score
    # Study Hours + Attendance + Sleep Quality
    
    # Map Sleep Quality
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    train_df['sleep_num'] = train_df['sleep_quality'].map(sleep_map)
    
    train_df['good_habits'] = (
        (train_df['study_hours'] / train_df['study_hours'].max()) + 
        (train_df['class_attendance'] / 100) + 
        (train_df['sleep_num'] / 2)
    )
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_df, x='Grade_Group', y='good_habits')
    plt.title('Good Habits Score by Grade Group')
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    img_dir = os.path.join(base_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    plt.savefig(os.path.join(img_dir, 'habits_vs_grade.png'))
    print(f"Saved boxplot to {os.path.join(img_dir, 'habits_vs_grade.png')}")
    
    # 3. Look at Quantiles
    # What % of 'A' students have attendance > 90?
    a_students = train_df[train_df['Grade_Group'] == 'A']
    f_students = train_df[train_df['Grade_Group'] == 'F']
    
    print(f"\n--- Extreme Statistics ---")
    print(f"Percentage of 'A' students with Attendance > 90%: {(a_students['class_attendance'] > 90).mean()*100:.1f}%")
    print(f"Percentage of 'F' students with Attendance < 70%: {(f_students['class_attendance'] < 70).mean()*100:.1f}%")
    
    print(f"Percentage of 'A' students with Study Hours > 20: {(a_students['study_hours'] > 20).mean()*100:.1f}%")
    
    # 4. Check Interactions
    # Is there a "Motivation" factor? (Study Method?)
    print("\n--- Study Method Distribution for 'A' Students ---")
    print(a_students['study_method'].value_counts(normalize=True))

if __name__ == "__main__":
    analyze_extremes()
