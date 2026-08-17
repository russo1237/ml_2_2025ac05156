import numpy as np
import pandas as pd
import os
import json
import joblib



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

data_path = "./model/heart.csv"

df = pd.read_csv(data_path)

# Data analysis
NUMERIC_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "HeartDisease"


print(FEATURE_COLUMNS)

print(f"Dataset shape: {df.shape[0]} instances, {df.shape[1]} raw feature columns(including target column)")
print(f"Class distribution:\n{df[TARGET_COL].value_counts()}\n")

X = df[FEATURE_COLUMNS]
y = df[TARGET_COL]

# 2. Train / test split (stratified)
#    The RAW test split (untouched, human-readable columns) is exactly
#    what gets written to test_data.csv, i.e. what a user uploads into
#    the Streamlit app.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"Saved test_data.csv with {test_df.shape[0]} rows -> ml_assign_2/test_data.csv")


# 3. Preprocessing: scale numeric columns, one-hot encode categoricals.
#    Fit ONLY on train, then reused identically for test/inference.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
)

print(preprocessor)

# 4. Define models. Each is wrapped with the SAME preprocessor inside a
#    Pipeline, so app.py only needs to load one .pkl per model and can
#    feed it the raw uploaded CSV directly -- no separate scaler/encoder
#    bookkeeping required at inference time.
model_defs = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42
    ),
}

# 5. Train, evaluate, save
results = {}

for name, base_model in model_defs.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", base_model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results[name] = metrics

    filename = "model/" + name.lower().replace(" ", "_").replace("-", "_") + ".pkl"
    joblib.dump(pipe, filename)
    print(f"Saved {filename}")

# 6. Save metadata the app needs
metadata = {
    "feature_columns": FEATURE_COLUMNS,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "target_column": TARGET_COL,
    "target_names": ["Normal", "HeartDisease"],  # index 0 -> Normal, 1 -> HeartDisease
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# 7. Print summary table for reference only
results_df = pd.DataFrame(results).T.round(4)
print("\n=== Model performance on held-out test set ===")
print(results_df)
results_df.to_csv("model/training_metrics_reference.csv")
print("\nSaved model/training_metrics_reference.csv")
