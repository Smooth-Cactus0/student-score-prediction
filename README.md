# Kaggle Playground Series S6E1: Student Exam Score Prediction

Repository for the Kaggle Playground Series S6E1 competition. The goal is to predict students' exam scores based on demographic and behavioral data using RMSE as the primary metric.

## Project Structure
- `data/`: Dataset files (train, test, sample submission).
- `scripts/`: Python scripts for EDA, preprocessing, and training.
- `submissions/`: Generated submission files.
- `images/`: Generated plots and visualizations.

## Benchmarks (Local CV RMSE)

| Version | Model | Features | CV RMSE |
|---------|-------|----------|---------|
| V1 | XGBoost | Baseline + Ordinal Encoding | 8.7610 |
| V2 | XGBoost | + Interactions, Target Encoding, Clustering | **8.7550** |
| V3 | Ensemble | XGB + LGBM + HistGB | 8.7644 |
| V4 | ANN | MLP (64, 32) | 8.8907 |
| V5 | XGBoost (Optuna) | V2 Features + Hyperparameter Tuning | 8.7564 |
| V6 | XGBoost | V2 + Course Relative Study, Interaction Features | 8.7554 |
| V4 Blend | Weighted Blend | 0.7*V2 + 0.2*V3 + 0.1*V4 | TBA |

## Key Findings
- **Study-Attendance Interaction**: The most powerful feature, accounting for ~47% of model importance.
- **Sleep Quality**: Significantly more impactful than sleep duration alone.
- **Behavioral Clustering**: Grouping students by habits helped the model distinguish between different student profiles.

## How to Run
1. Create virtual environment: `python -m venv .venv`
2. Activate and install deps: `.venv\Scripts\python.exe -m pip install -r requirements.txt` (TODO: generate requirements.txt)
3. Run training: `.venv\Scripts\python.exe scripts/train_v2.py`
