import pandas as pd
import os
import sys

def blend_final():
    print("Loading submissions for Final Blend...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    
    try:
        sub_pseudo = pd.read_csv(os.path.join(sub_dir, 'submission_v8_pseudo.csv')) # 8.7343
        sub_tree = pd.read_csv(os.path.join(sub_dir, 'submission_ensemble_v3.csv')) # 8.7644
        sub_ann = pd.read_csv(os.path.join(sub_dir, 'submission_ann.csv')) # 8.8907
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Blending...")
    # Weights: 85% Pseudo, 10% Tree Ensemble, 5% ANN
    
    blend_preds = (
        0.85 * sub_pseudo['exam_score'] + 
        0.10 * sub_tree['exam_score'] + 
        0.05 * sub_ann['exam_score']
    )
    
    submission = sub_pseudo.copy()
    submission['exam_score'] = blend_preds
    
    save_path = os.path.join(sub_dir, 'submission_v9_final_blend.csv')
    submission.to_csv(save_path, index=False)
    print(f"Final Blended submission saved to '{save_path}'")
    print("Weights: 0.85 Pseudo-XGB + 0.10 TreeEns + 0.05 ANN")

if __name__ == "__main__":
    blend_final()
