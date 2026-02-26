# app.py
import streamlit as st
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# App setup
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
st.title("📞 Telco Customer Churn Prediction")
st.markdown("**BITS Pilani - Assignment 2** | Model: SMOTE Enhanced")

# Load model artifacts

MODEL_PATH = "model/outputModel"

if not os.path.exists(MODEL_PATH):
    st.error(f"Folder '{MODEL_PATH}' not found. Please run the notebook first.")
    st.stop()


@st.cache_resource
def load_resources():
    # Return (None, None) if artifacts are missing.
    try:
        scaler = pickle.load(open(f"{MODEL_PATH}/scaler.pkl", "rb"))
        feature_names = pickle.load(open(f"{MODEL_PATH}/features.pkl", "rb"))
        return scaler, feature_names
    except FileNotFoundError:
        return None, None


def preprocess_data(df, scaler, feature_names):
    # Match training-time feature columns before scaling.
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    return scaler.transform(df_encoded)


@st.cache_resource
def load_model(path):
    return pickle.load(open(path, "rb"))


scaler, feature_names = load_resources()

if not scaler:
    st.error(
        "Artifacts missing. Run your training notebook to generate 'scaler.pkl' and 'features.pkl'."
    )
    st.stop()

# Single-customer input
st.sidebar.header("Customer Profile")


def user_input_features():
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
    # Default estimate; user can overwrite.
    total_charges = st.sidebar.number_input(
        "Total Charges ($)", 0.0, 10000.0, tenure * monthly_charges
    )

    contract = st.sidebar.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    internet_service = st.sidebar.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        # Fixed defaults to keep the demo form small.
        "gender": "Male",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "PaperlessBilling": "Yes",
    }
    return pd.DataFrame(data, index=[0])


input_df = user_input_features()

# Prediction
st.subheader("Prediction Interface")
model_list = [
    f
    for f in os.listdir(MODEL_PATH)
    if f.endswith(".pkl") and f not in ["scaler.pkl", "features.pkl"]
]
selected_model = st.selectbox("Select Model", model_list)

if st.button("Predict Churn"):
    input_scaled = preprocess_data(input_df, scaler, feature_names)

    model = load_model(f"{MODEL_PATH}/{selected_model}")

    pred = model.predict(input_scaled)[0]
    prob = (
        model.predict_proba(input_scaled)[0][1]
        if hasattr(model, "predict_proba")
        else 0
    )

    if pred == 1:
        st.error(f"⚠️ High Churn Risk (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Safe Customer (Probability: {prob:.2%})")

# Batch prediction
st.divider()
st.subheader("Batch Prediction (Upload CSV)")
sample_test_path = "model/data/processed/test.csv"
if os.path.exists(sample_test_path):
    with open(sample_test_path, "rb") as f:
        st.download_button(
            label="Download Test CSV",
            data=f.read(),
            file_name="test.csv",
            mime="text/csv",
        )
else:
    st.warning(f"Sample file not found at '{sample_test_path}'")

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    test_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", test_data.head(3))

    if st.button("Run Batch Prediction"):
        if "customerID" in test_data.columns:
            test_data = test_data.drop("customerID", axis=1)

        if "TotalCharges" in test_data.columns:
            test_data["TotalCharges"] = pd.to_numeric(
                test_data["TotalCharges"], errors="coerce"
            ).fillna(0)

        # Split labels if present.
        if "Churn" in test_data.columns:
            y_true = test_data["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
            X_test = test_data.drop("Churn", axis=1)
        else:
            y_true = None
            X_test = test_data

        X_test_scaled = preprocess_data(X_test, scaler, feature_names)

        model = load_model(f"{MODEL_PATH}/{selected_model}")
        y_pred = model.predict(X_test_scaled)

        st.success("Predictions generated!")
        test_data["Prediction"] = y_pred
        st.dataframe(test_data[["Prediction"]].head())

        # Show metrics only when ground truth exists.
        if y_true is not None:
            st.divider()
            st.subheader("📊 Model Performance Report")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
            m2.metric("Precision", f"{precision_score(y_true, y_pred):.2%}")
            m3.metric("Recall", f"{recall_score(y_true, y_pred):.2%}")
            m4.metric("F1 Score", f"{f1_score(y_true, y_pred):.2%}")

            col_left, col_right = st.columns(2)

            with col_left:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with col_right:
                st.write("**Classification Report**")
                report = classification_report(y_true, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
        else:
            st.info(
                "Note: Upload a CSV with a 'Churn' column to see evaluation metrics."
            )
