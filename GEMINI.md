# Playground Series S6E1 - Student Exam Score Prediction

## Project Overview

This directory contains the dataset for the **[Playground Series Season 6, Episode 1](https://www.kaggle.com/competitions/playground-series-s6e1/overview)** competition.

**Context:** The dataset is synthetically generated from a deep learning model trained on a real-world dataset. The task is to predict the test scores of students given their demographic and academic details.

**Goal:** Build the best possible regression model (or ensemble of models) to achieve the top ranking by the end of the competition.

**Evaluation Metric:** Submissions are evaluated on the **Root Mean Squared Error (RMSE)** between the predicted and observed `exam_score`.

## Dataset Description

The dataset is split into training and testing sets.

### Files

*   **`train.csv`**: The training dataset. Contains features and the target variable `exam_score`.
*   **`test.csv`**: The test dataset. Contains features only. Use this to generate predictions.
*   **`sample_submission.csv`**: A sample submission file showing the correct format for uploading predictions.

### Features

The dataset includes the following features:

*   **`id`**: Unique identifier for each record.
*   **`age`**: Student's age.
*   **`gender`**: Student's gender.
*   **`course`**: The course the student is enrolled in.
*   **`study_hours`**: Number of hours studied.
*   **`class_attendance`**: Percentage of class attendance.
*   **`internet_access`**: Whether the student has internet access (`yes`/`no`).
*   **`sleep_hours`**: Average hours of sleep.
*   **`sleep_quality`**: Quality of sleep (e.g., `poor`, `average`, `good`).
*   **`study_method`**: Method of study (e.g., `group study`, `online videos`, `self-study`).
*   **`facility_rating`**: Rating of the educational facilities.
*   **`exam_difficulty`**: Perceived difficulty of the exam.

### Target Variable

*   **`exam_score`**: The score achieved by the student (found in `train.csv`).

## Data Statistics

Based on initial analysis of `train.csv` (630,000 records):
- **Average Score**: ~62.51
- **Minimum Score**: ~19.60
- **Maximum Score**: 100.0

## Usage Tips

When using the Gemini CLI with this dataset, it is recommended to increase the maximum file size limit to handle the large CSV files:

```bash
gemini --max-file-size 100MB
```

## Getting Started

To begin working on this project:

1.  **Data Exploration:** Load `train.csv` using a library like `pandas` to analyze distributions and correlations.
2.  **Preprocessing:** Handle any missing values, encode categorical variables (like `gender`, `course`, `sleep_quality`), and scale numerical features if necessary.
3.  **Modeling:** Train regression models (e.g., Linear Regression, Random Forest, XGBoost) on the data.
4.  **Prediction:** Generate predictions for the rows in `test.csv`.
5.  **Submission:** Format your predictions according to `sample_submission.csv`.

## Tools

Recommended tools for this analysis:
*   **Python**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`.
*   **Jupyter Notebooks**: For interactive data analysis.
