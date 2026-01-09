import pandas as pd
import os
import sys

def blend_grandmaster():
    print("Loading submissions for Grandmaster Blend...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    
    try:
        sub_ultimate = pd.read_csv(os.path.join(sub_dir, 'submission_v11_ultimate.csv')) # 8.7294
        sub_tree = pd.read_csv(os.path.join(sub_dir, 'submission_ensemble_v3.csv')) # 8.7644
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Blending...")
    # Weights: 90% Ultimate, 10% Tree Ensemble
    
    blend_preds = (
        0.90 * sub_ultimate['exam_score'] + 
        0.10 * sub_tree['exam_score']
    )
    
    submission = sub_ultimate.copy()
    submission['exam_score'] = blend_preds
    
    save_path = os.path.join(sub_dir, 'submission_v12_grandmaster.csv')
    submission.to_csv(save_path, index=False)
    print(f"Grandmaster Blended submission saved to '{save_path}'")
    print("Weights: 0.90 Ultimate (V11) + 0.10 TreeEns (V3)")

if __name__ == "__main__":
    blend_grandmaster()
