# telco-churn-eda-project
# Customer Churn Analysis: EDA, Visualization and Deep Learning

Course: U21ADP05 - Exploratory Data Analysis and Visualization (Assignment II)

Student: Sri Sivaranjani M, 24AD115, AD - B (III), Dept. of AI and Data Science, KPR Institute of Engineering and Technology

## Overview
This project analyses the IBM Telco Customer Churn dataset (7,043 customers, 21 features) to find
why customers leave a telecom company, and builds an MLP deep learning model to predict churn.

## Dataset
Telco Customer Churn (IBM sample data), Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## What the project covers
- Data cleaning (blank TotalCharges, duplicates, outliers), encoding, scaling, 70/15/15 split
- 7 visualizations covering the seven stages of visualization
- Statistical analysis: normality test, Binomial fit, chi-square test, t-test, correlation
- Feature analysis and permutation feature importance
- MLP model (128 units, dropout 0.4, learning rate 0.001) with basic hyperparameter tuning

## Key results
- Test accuracy: 74.6%
- Churn recall: 0.80, precision: 0.51
- AUC: 0.839
- Most influential factors: tenure, contract type, monthly charges

## Repository structure
- `Source_code.ipynb`: complete code with outputs
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: dataset
- `EDA_Report.pdf`: full project report
- `README.md`: project description

## How to run
1. Open `Source_code.ipynb` in Google Colab.
2. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to the Colab files area (`/content`).
3. Run all cells (Runtime → Run all).

Libraries: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, tensorflow.
