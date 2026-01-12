import pandas as pd
import os
import sys

def blend_v14():
    print("Loading submissions for Blend V14...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_dir = os.path.join(base_dir, 'submissions')
    
    try:
        sub_teacher = pd.read_csv(os.path.join(sub_dir, 'submission_v13_teacher.csv'))
        sub_lgbm = pd.read_csv(os.path.join(sub_dir, 'submission_lgbm.csv')) # 8.7636
        sub_ann = pd.read_csv(os.path.join(sub_dir, 'submission_ann.csv'))   # 8.8907
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("Blending...")
    # Weights: 70% Teacher (XGB), 25% LightGBM, 5% ANN
    # Rationale: Teacher is strongest (distilled), LGBM adds tree diversity, ANN adds non-tree diversity.
    
    blend_preds = (
        0.70 * sub_teacher['exam_score'] + 
        0.25 * sub_lgbm['exam_score'] + 
        0.05 * sub_ann['exam_score']
    )
    
    submission = sub_teacher.copy()
    submission['exam_score'] = blend_preds
    
    save_path = os.path.join(sub_dir, 'submission_v14_blend.csv')
    submission.to_csv(save_path, index=False)
    print(f"V14 Blended submission saved to '{save_path}'")
    print("Weights: 0.70 Teacher(V13) + 0.25 LGBM + 0.05 ANN")

if __name__ == "__main__":
    blend_v14()
