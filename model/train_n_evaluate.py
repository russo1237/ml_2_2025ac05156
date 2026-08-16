import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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
# ---------------------------------------------------------------------
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






