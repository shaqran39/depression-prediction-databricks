# Depression Dataset — ML Pipeline on Databricks

Predicting **history of mental illness** from lifestyle and socio-economic factors, using a Kaggle dataset processed end-to-end on Databricks (PySpark + pandas + scikit-learn + XGBoost).

## Overview

This project explores whether lifestyle and socio-economic factors (income, employment, sleep, diet, marital status, family history, etc.) can predict whether an individual has a recorded history of mental illness. It covers the full pipeline: ingestion, cleaning, exploratory analysis, correlation analysis, model tuning and comparison, and an interactive prediction tool.

**Key finding:** none of the 14 available features showed more than a weak correlation with the target. As a result, the simplest model tested (Logistic Regression) outperformed both tree-based ensembles (Random Forest, XGBoost) — a good reminder that model complexity only helps when there's non-linear structure in the data for it to find.

## Dataset

- **Source:** [Kaggle — Depression Dataset](https://www.kaggle.com/datasets/anthonytherrien/depression-dataset) by anthonytherrien
- **Size:** 413,768 records, 16 columns, no missing values, no duplicates
- **Target:** `History of Mental Illness` (Yes/No) — the closest available proxy to a depression label, since the dataset has no depression-specific diagnosis column
- **Features:** Age, Marital Status, Education Level, Number of Children, Smoking Status, Physical Activity Level, Employment Status, Income, Alcohol Consumption, Dietary Habits, Sleep Patterns, History of Substance Abuse, Family History of Depression, Chronic Medical Conditions

## Tech Stack

| Layer | Tools |
|---|---|
| Ingestion & storage | Databricks, Unity Catalog Volumes |
| Data processing | PySpark |
| Analysis & visualisation | pandas, seaborn, matplotlib |
| Modelling | scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Interactive demo | Databricks `dbutils.widgets` |

## Pipeline

1. **Ingestion** — raw CSV uploaded to a Unity Catalog Volume, loaded into a Spark DataFrame
2. **Cleaning** — type casting (Age, Income, Number of Children to numeric), null/duplicate checks, categorical value validation, outlier checks — dataset came back clean on all counts
3. **EDA** — distribution plots, categorical breakdowns vs. target, Age/Income density, income-by-employment boxplots, family history cross-tab
4. **Correlation analysis** — label-encoded Pearson correlation of every feature against the target
5. **Modelling** — three algorithms, each tuned across two independent hyperparameter searches (5+ params each), compared on held-out test F1
6. **Evaluation** — confusion matrix, classification report, ROC/AUC for the winning model
7. **Interactive widget** — notebook-native input panel for live what-if predictions

## Results

| Model | Test F1 (Yes class) | Precision | Recall | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.445** | 0.36 | 0.58 | 0.56 |
| Random Forest | 0.432 | 0.38 | 0.50 | 0.60 |
| XGBoost | 0.123 | 0.37 | 0.07 | 0.68 |

Logistic Regression was selected as the final model. XGBoost had the highest raw accuracy but mostly defaulted to predicting "No," a classic symptom of optimising for accuracy on imbalanced data rather than the metric that actually matters here.

**Final model performance:** F1 = 0.445, AUC = 0.59 (only modestly better than random guessing). This model should be read as a weak, directional signal rather than a diagnostic tool.

## Correlation Snapshot

Strongest (still weak) relationships with the target:
- Employment Status: **+0.14**
- Income: **−0.14**
- Everything else (Age, Sleep, Diet, Marital Status, Family History of Depression, etc.): **between −0.02 and +0.03**

## Interactive Prediction Widget

Built with `dbutils.widgets` directly in the notebook — enter a hypothetical person's Age, Income, Employment Status, etc. via dropdowns/text boxes, and get a live prediction plus the model's confidence score. Useful for demonstrating the model's uncertainty rather than just a static Yes/No.

## Limitations

- Target is general "history of mental illness," not a depression-specific diagnosis
- The uniform Age distribution suggests parts of the dataset may be synthetically generated
- Label encoding of categoricals may understate genuinely non-linear relationships
- Not suitable for standalone diagnostic or high-stakes decision-making — recommended as a screening signal paired with human review only

## Full Report

A complete write-up with all charts, code, and discussion is available in this repo: [`Depression_Dataset_ML_Handover_Report.docx`](./Depression_Dataset_ML_Handover_Report.docx)

## Author

**Shaqran Bin Saleh** — [LinkedIn](https://linkedin.com/in/shaqran39) · [GitHub](https://github.com/shaqran39)
