# Telco Customer Churn Prediction
**BITS Pilani - Assignment 2**

## 1. Problem Statement
To predict whether a telecommunications customer will churn (cancel service) based on their demographics, usage, and account information. This helps the company target retention offers effectively.

## 2. Dataset Description
* **Source:** Kaggle (Telco Customer Churn)
* **Instances:** 7043
* **Features:** 21 (including Target)
* **Key Features:** Tenure, Monthly Charges, Contract Type, Payment Method.

## 3. Model Comparison Table
| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.756 | 0.861 | 0.524 | 0.834 | 0.644 | 0.502 |
| Decision Tree | 0.745 | 0.847 | 0.512 | 0.788 | 0.621 | 0.465 |
| KNN | 0.709 | 0.816 | 0.471 | 0.812 | 0.596 | 0.429 |
| Naive Bayes | 0.668 | 0.838 | 0.438 | 0.893 | 0.587 | 0.425 |
| Random Forest | 0.791 | 0.856 | 0.587 | 0.713 | 0.644 | 0.503 |
| XGBoost | 0.799 | 0.860 | 0.604 | 0.710 | 0.653 | 0.517 |


## 4. Observations
* **Best Overall Model:** XGBoost achieved the highest Accuracy (0.799), highest F1 Score (0.653), and highest MCC (0.517).
* **Recall vs. Precision Trade-off:** Naive Bayes had the highest Recall (0.893) but the lowest Precision (0.438), while XGBoost had the highest Precision (0.604) with Recall of 0.710.
* **AUC Leader:** Logistic Regression recorded the highest AUC (0.861), very close to XGBoost (0.860).
* **Strong Alternatives:** Random Forest and Logistic Regression both achieved F1 Score of 0.644, but Random Forest had higher Accuracy (0.791 vs 0.756).
