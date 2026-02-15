# Telco Customer Churn Prediction
### BITS Pilani - Assignment 2

## 1. Problem Statement
To predict whether a telecommunications customer will churn (cancel service) based on their demographics, usage, and account information. This helps the company target retention offers effectively.

## 2. Dataset Description
* **Source:** Kaggle (Telco Customer Churn)
* **Instances:** 7043
* **Features:** 21 (including Target)
* **Key Features:** Tenure, Monthly Charges, Contract Type, Payment Method.

## 3. Model Comparison Table
The following table compares the performance of various models after applying SMOTE for class balancing:

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.756 | 0.861 | 0.524 | 0.834 | 0.644 | 0.502 |
| **Decision Tree** | 0.745 | 0.847 | 0.512 | 0.788 | 0.621 | 0.465 |
| **KNN** | 0.709 | 0.816 | 0.471 | 0.812 | 0.596 | 0.429 |
| **Naive Bayes** | 0.668 | 0.838 | 0.438 | 0.893 | 0.587 | 0.425 |
| **Random Forest** | 0.791 | 0.856 | 0.587 | 0.713 | 0.644 | 0.503 |
| **XGBoost** | 0.799 | 0.860 | 0.604 | 0.710 | 0.653 | 0.517 |

## 4. Observations
Below are the specific observations regarding the performance of each model on this dataset:

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** |Performed well with a high Recall (83%), making it effective for identifying potential churners, though it had a moderate false positive rate. |
| **Decision Tree** | Provided decent baseline performance but struggled with precision compared to ensemble methods, likely due to overfitting on the minority class. |
| **KNN** | Achieved the lowest accuracy (70%) and precision among the group; performance was likely impacted by the high dimensionality and noise in the data. |
| **Naive Bayes** | Achieved the highest Recall (89%) of all models but the lowest Accuracy (66%) and Precision, indicating it flags many non-churners as churners (high False Positives). |
| **Random Forest** | Significantly improved accuracy (79%) and precision over the single Decision Tree by reducing variance, providing a stable and robust prediction. |
| **XGBoost** | **Best Performing Model.** Achieved the highest overall Accuracy (approx 80%) and F1 Score (0.65), offering the best balance between Precision and Recall. |