a. Problem statement - Predict the presence or absence of heart disease in a patient based on clinical parameters and diagnostic features.

b. Dataset description - 
    Dataset Name: Heart Failure Prediction Dataset
    Features(P): 12 numeric and categorical features
    Instances(N): 918 rows
    Target Variable: HeartDisease (Binary: 1 = Heart Disease present, 0 = Normal)

c. Github Repository Link - https://github.com/russo1237/ml_2_2025ac05156

d. Comparison table

### Model Performance on Held-Out Test Set

| ML Model name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8859 | 0.9299 | 0.8716 | 0.9314 | 0.9005 | 0.7694 |
| **Decision Tree** | 0.7989 | 0.8440 | 0.8218 | 0.8137 | 0.8177 | 0.5935 |
| **K-Nearest Neighbors** | **0.9185** | **0.9507** | **0.9223** | **0.9314** | **0.9268** | **0.8349** |
| **Naive Bayes** | 0.8859 | 0.9118 | 0.8932 | 0.9020 | 0.8976 | 0.7688 |
| **Random Forest** | 0.9022 | 0.9321 | 0.8962 | 0.9314 | 0.9135 | 0.8018 |



Observations :

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Demonstrates strong and balanced performance with high Recall (0.9314) and solid AUC (0.9299), showing effective linear separation capability. |
| **Decision Tree** | Lowest overall performance across all metrics (Accuracy: 0.7989, MCC: 0.5935), likely suffering from variance/overfitting on the training distribution. |
| **kNN** | Top performer across the board; achieved the highest Accuracy (0.9185), AUC (0.9507), Precision (0.9223), F1 Score (0.9268), and MCC (0.8349). |
| **Naive Bayes** | Solid baseline performance with a balanced Precision (0.8932) and Recall (0.9020), though slightly trailing behind the tree ensemble and kNN models. |
| **Random Forest (Ensemble)** | Second-best overall model; significantly improves over a single Decision Tree across all metrics (Accuracy: 0.9022, AUC: 0.9321, F1: 0.9135). |
| **Overall Winner for your dataset?** | **kNN (K-Nearest Neighbors)** is the clear winner, consistently outperforming all other models across discrimination (AUC: 0.9507), general accuracy (91.85%), and correlation balance (MCC: 0.8349). |
