import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.sidebar.title("Heart Disease App")

# Download Dataset
MODEL_DIR = "model"
DATA_PATH = "data/HeartDisease_test.csv"
st.sidebar.subheader("Download Dataset")

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "rb") as f:
        st.sidebar.download_button(
            label="HeartDisease_test.csv",
            data=f,
            file_name="HeartDisease_test.csv",
            mime="text/csv"
        )
else:
    st.sidebar.warning("Dataset not found in data/ folder.")

# Upload Dataset
st.sidebar.subheader("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload HeartDisease_test.csv",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(DATA_PATH)

st.title("Heart Disease Prediction Model")
st.write("Dataset Preview:")
st.dataframe(df.head())

# Model selection
st.subheader("Select Model")

available_models = {
    "Logistic Regression": "Logistic_Regression.pkl",
    "Decision Tree": "Decision_Tree.pkl",
    "KNN": "KNN.pkl",
    "Naive Bayes": "Naive_Bayes.pkl",
    "Random Forest": "Random_Forest.pkl",
    "XGBoost": "XGBoost.pkl"
}

model_name = st.selectbox("Choose a model", list(available_models.keys()))
model_path = os.path.join(MODEL_DIR, available_models[model_name])

# Load model
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = joblib.load(model_path)

# Prepare data
TARGET = "HeartDisease"
if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' not found in dataset.")
    st.stop()

X = df.drop(columns=[TARGET])
y = df[TARGET]
preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.pkl'))
X = preprocessor.transform(X)
X = pd.DataFrame(X, columns = preprocessor.get_feature_names_out())

# Predictions
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Metrics Table
st.subheader("Evaluation Metrics")

metrics = {
    "Accuracy": accuracy_score(y, y_pred),
    "AUC": roc_auc_score(y, y_prob),
    "Precision": precision_score(y, y_pred),
    "Recall": recall_score(y, y_pred),
    "F1 Score": f1_score(y, y_pred),
    "MCC": matthews_corrcoef(y, y_pred),
}

metrics_df = pd.DataFrame(metrics, index=["Score"]).T.round(2)
st.dataframe(metrics_df)

col1, col2 = st.columns(2)

# Confusion Matrix
with col1:
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(2, 2))

    ax.imshow(cm)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)
    ax.set_title("Confusion Matrix", fontsize=7)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=7)
    ax.set_yticklabels(["0", "1"], fontsize=7)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)

    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)

# Classification Report
with col2:
    st.subheader("Classification Report")

    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(2)

    st.dataframe(report_df, use_container_width=True)