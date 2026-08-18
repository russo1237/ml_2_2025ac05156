import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

st.set_page_config(page_title="ML assign 2 : 2025AC05156@wilp.bits-pilani.ac.in :Heart Failure Predictor", layout="wide")

# 1. Load artifacts
@st.cache_resource
def load_models():
    with open("model/metadata.json") as f:
        meta = json.load(f)
    model_paths = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "KNN": "model/k_nearest_neighbors.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest": "model/random_forest.pkl",
    }
    models = {name: joblib.load(path) for name, path in model_paths.items()}
    return meta, models

meta, models = load_models()
features = meta["feature_columns"]
target = meta["target_column"]
target_names = meta["target_names"]

# 2. Sidebar controls
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox("Choose Model", list(models.keys()))

st.title("ML assign 2 : 2025AC05156@wilp.bits-pilani.ac.in")
st.title("Heart Failure Predictor")


# 3. Main-page upload interface
st.subheader("📥 Upload Test Data")
st.caption(
    "**Note:** Test data should be a CSV with columns: `Age`, `Sex`, `ChestPainType`, "
    "`RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, "
    "`Oldpeak`, `ST_Slope`, and optionally `HeartDisease`."
)
st.subheader("📥 Sample test_data.csv is available here (You can download zip and upload the test_data.csv) : https://github.com/russo1237/ml_2_2025ac05156")

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

# 4. Results execution — only runs when a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validate required columns
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    st.markdown("---")
    st.write(f"Scoring **{len(df)}** records with **{selected_model}**")

    X = df[features]
    pipe = models[selected_model]
    y_pred = pipe.predict(X)
    
    # Handle single-probability vs multi-class array safely
    proba = pipe.predict_proba(X)
    y_proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba

    # Evaluation Matrix Section (Requires Ground Truth Target)
    if target in df.columns:
        y_true = df[target]

        st.subheader("Model Evaluation Matrix")

        # 1. Calculate Aggregate Metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = None

        # 2. Display KPI Cards
        cols = st.columns(6 if auc is not None else 5)
        cols[0].metric("Accuracy", f"{acc:.4f}")
        cols[1].metric("Precision", f"{prec:.4f}")
        cols[2].metric("Recall", f"{rec:.4f}")
        cols[3].metric("F1 Score", f"{f1:.4f}")
        cols[4].metric("MCC", f"{mcc:.4f}")
        if auc is not None:
            cols[5].metric("ROC-AUC", f"{auc:.4f}")

        # 3. Visual & Tabular Breakdowns
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(
                cm,
                index=[f"Actual: {name}" for name in target_names],
                columns=[f"Predicted: {name}" for name in target_names]
            )
            st.dataframe(cm_df, use_container_width=True)

        with col_right:
            st.markdown("**Classification Report**")
            report = classification_report(
                y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
    else:
        st.warning(f"Target column '{target}' not detected in CSV. Ground-truth evaluation matrix cannot be calculated.")

    # --- Sample Predictions (First 10 Entries) ---
    st.markdown("---")
    st.subheader("Sample Predictions Preview (First 10 Records)")

    pred_sample_df = df.head(10).copy()
    
    # Map predictions to class names
    pred_sample_df["Predicted_Class"] = [target_names[p] if isinstance(p, (int, np.integer)) and p < len(target_names) else p for p in y_pred[:10]]
    
    # Add predicted probability
    pred_sample_df["Prediction_Probability"] = np.round(y_proba[:10], 4)

    # If actual labels exist, show match status
    if target in df.columns:
        pred_sample_df["Actual_Class"] = [target_names[a] if isinstance(a, (int, np.integer)) and a < len(target_names) else a for a in y_true[:10]]
        pred_sample_df["Is_Correct"] = pred_sample_df["Predicted_Class"] == pred_sample_df["Actual_Class"]

    st.dataframe(pred_sample_df, use_container_width=True)

else:
    st.info("👆 Please upload a test CSV file above to start model scoring and choose a model from left sidebar")