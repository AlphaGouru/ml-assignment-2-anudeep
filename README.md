# Machine Learning Assignment 2

## Problem Statement
The objective of this assignment is to implement multiple machine learning classification models on a marketing dataset and compare their performance using standard evaluation metrics. An interactive Streamlit web application is also developed to demonstrate predictions and model performance through a user-friendly interface.

---

## Dataset Description
The Bank Marketing dataset from the UCI Machine Learning Repository is used in this assignment. The dataset contains over 45,000 customer records with demographic and marketing interaction attributes. The goal is to predict whether a customer subscribes to a term deposit based on various features. The dataset contains more than 12 input features and satisfies assignment requirements for classification tasks.

---

## Models Used and Evaluation Metrics Comparison

The following six machine learning classification models are implemented and evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.884 | 0.848 | 0.551 | 0.202 | 0.296 | 0.285 |
| Decision Tree | 0.870 | 0.691 | 0.459 | 0.455 | 0.457 | 0.383 |
| kNN | 0.877 | 0.756 | 0.481 | 0.266 | 0.342 | 0.296 |
| Naive Bayes | 0.836 | 0.812 | 0.361 | 0.466 | 0.407 | 0.317 |
| Random Forest (Ensemble) | 0.904 | 0.924 | 0.657 | 0.420 | 0.512 | 0.476 |
| XGBoost (Ensemble) | 0.904 | 0.928 | 0.624 | 0.507 | 0.560 | 0.510 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Provides a strong baseline but struggles to capture complex patterns, resulting in lower recall. |
| Decision Tree | Performs better than baseline but shows slight overfitting due to model complexity. |
| kNN | Sensitive to data distribution and distance metrics, resulting in moderate performance. |
| Naive Bayes | Performs reasonably well despite independence assumptions among features. |
| Random Forest (Ensemble) | Improves performance by combining multiple decision trees and reduces overfitting. |
| XGBoost (Ensemble) | Achieves the best overall performance due to boosting and advanced optimization techniques. |

---

## Streamlit Application Features
The deployed Streamlit web application provides:
- CSV test dataset upload option
- Model selection dropdown
- Prediction and evaluation display
- Classification report output
- Confusion matrix visualization

---

## Repository Structure
ml-assignment-2/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│ └── bank_marketing.csv
│
└── models/
├── train_models.py
├── model_metrics.csv
└── saved model files (.pkl)


---

## Deployment
The application is deployed using **Streamlit Community Cloud** and provides an interactive interface for evaluation.

---

## Conclusion
This project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, comparison, web app development, and deployment for real-world usage.
