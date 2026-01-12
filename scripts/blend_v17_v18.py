import pandas as pd
import numpy as np
import os

def blend_strategies():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    
    # Load Models
    # V14: Best Global (8.705 on LB)
    try:
        sub_v14 = pd.read_csv(os.path.join(sub_dir, 'submission_v14_blend.csv'))
    except FileNotFoundError:
        print("V14 not found. Please run blend_v14.py first.")
        return

    # V16: Best Tails (Augmented)
    try:
        sub_v16 = pd.read_csv(os.path.join(sub_dir, 'submission_v16_aug.csv'))
    except FileNotFoundError:
        print("V16 not found. Please run train_v16_aug.py first.")
        return

    print("Generating V17: Simple Linear Blend (80% V14 + 20% V16)...")
    v17_preds = 0.8 * sub_v14['exam_score'] + 0.2 * sub_v16['exam_score']
    
    sub_v17 = sub_v14.copy()
    sub_v17['exam_score'] = v17_preds
    sub_v17.to_csv(os.path.join(sub_dir, 'submission_v17_linear_aug.csv'), index=False)
    print("V17 Saved.")

    print("Generating V18: Smart Conditional Blend...")
    # Idea: Trust V16 more at the extremes.
    # We use the mean prediction to decide where we are.
    
    mean_preds = (sub_v14['exam_score'] + sub_v16['exam_score']) / 2
    
    # Weights array
    # Default: 90% V14 (Safety)
    w_v14 = np.ones(len(mean_preds)) * 0.90
    
    # If prediction > 85 (High), trust V16 more (e.g., 50/50)
    # The higher it goes, the more we trust V16
    # Smooth transition?
    # Let's simple threshold for now.
    
    # High zone
    mask_high = mean_preds > 85
    w_v14[mask_high] = 0.50 # 50% split at top
    
    # Low zone
    mask_low = mean_preds < 60
    w_v14[mask_low] = 0.50 # 50% split at bottom
    
    w_v16 = 1.0 - w_v14
    
    v18_preds = w_v14 * sub_v14['exam_score'] + w_v16 * sub_v16['exam_score']
    
    sub_v18 = sub_v14.copy()
    sub_v18['exam_score'] = v18_preds
    sub_v18.to_csv(os.path.join(sub_dir, 'submission_v18_smart_aug.csv'), index=False)
    print("V18 Saved.")
    
    print("\nSummary:")
    print("V17: 0.8*V14 + 0.2*V16 (Conservative Global Improvement)")
    print("V18: Dynamic weights (50/50 at extremes, 90/10 in middle)")

if __name__ == "__main__":
    blend_strategies()
