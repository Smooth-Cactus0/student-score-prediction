import pandas as pd
import os

def blend_predictions():
    print("Loading submissions...")
    # Using relative paths assuming run from root
    try:
        sub_xgb = pd.read_csv('submissions/submission_v2.csv') # Best single model (8.755)
        sub_tree = pd.read_csv('submissions/submission_ensemble_v3.csv') # Tree Ensemble (8.764)
        sub_ann = pd.read_csv('submissions/submission_ann.csv') # ANN (8.891)
    except FileNotFoundError:
        print("Error: One or more submission files not found. Run previous steps first.")
        return

    print("Blending...")
    # Weighted Average
    # XGB is dominant. ANN is weak but maybe orthogonal.
    # Weights: 70% XGB, 20% Tree Ens, 10% ANN
    
    blend_preds = (
        0.7 * sub_xgb['exam_score'] + 
        0.2 * sub_tree['exam_score'] + 
        0.1 * sub_ann['exam_score']
    )
    
    submission = sub_xgb.copy()
    submission['exam_score'] = blend_preds
    
    submission.to_csv('submissions/submission_blend_v4.csv', index=False)
    print(f"Blended submission saved to 'submissions/submission_blend_v4.csv'")
    print("Blend Weights: 0.7 XGB + 0.2 TreeEns + 0.1 ANN")

if __name__ == "__main__":
    blend_predictions()
