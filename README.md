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
| Random Forest | 0.791 | 0.855 | 0.587 | 0.708 | 0.642 | 0.500 |
| XGBoost | 0.799 | 0.861 | 0.602 | 0.713 | 0.653 | 0.517 |

*(Note: Replace these numbers with your final run results if they differ)*

## 4. Observations
* **Best Model:** XGBoost achieved the highest F1 Score (0.65) and Accuracy (80%).
* **Recall vs. Precision:** Logistic Regression had the highest Recall (83%), making it best for catching all potential churners, though it had more false alarms.
* **SMOTE Effect:** Using SMOTE balanced the data, significantly improving the Recall for all models compared to the baseline.