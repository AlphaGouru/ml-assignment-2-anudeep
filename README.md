\# Machine Learning Assignment 2



\## Problem Statement

The objective of this assignment is to build multiple machine learning classification models on a marketing dataset and compare their performance using standard evaluation metrics. An interactive Streamlit application is also developed to demonstrate model predictions.



---



\## Dataset Description

The Bank Marketing dataset from the UCI Machine Learning Repository is used. It contains over 45,000 records with customer demographic and marketing interaction information. The task is to predict whether a customer subscribes to a term deposit based on multiple attributes. The dataset contains more than 12 features and satisfies assignment requirements.



---



\## Models Used and Evaluation Metrics Comparison



| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |

|---------------|----------|-----|-----------|--------|----|-----|

| Logistic Regression | 0.884 | 0.848 | 0.551 | 0.202 | 0.296 | 0.285 |

| Decision Tree | 0.870 | 0.691 | 0.459 | 0.455 | 0.457 | 0.383 |

| kNN | 0.877 | 0.756 | 0.481 | 0.266 | 0.342 | 0.296 |

| Naive Bayes | 0.836 | 0.812 | 0.361 | 0.466 | 0.407 | 0.317 |

| Random Forest (Ensemble) | 0.904 | 0.924 | 0.657 | 0.420 | 0.512 | 0.476 |

| XGBoost (Ensemble) | 0.904 | 0.928 | 0.624 | 0.507 | 0.560 | 0.510 |



---



\## Observations on Model Performance



| ML Model Name | Observation about model performance |

|----------------|------------------------------------|

| Logistic Regression | Provides a stable baseline but struggles to capture complex patterns, resulting in lower recall. |

| Decision Tree | Performs better than baseline but shows signs of overfitting due to model complexity. |

| kNN | Sensitive to data distribution and distance metrics, giving moderate performance. |

| Naive Bayes | Performs reasonably well despite strong independence assumptions among features. |

| Random Forest (Ensemble) | Improves performance by combining multiple trees and reduces overfitting compared to a single tree. |

| XGBoost (Ensemble) | Achieves the best overall performance due to boosting and optimization techniques. |



