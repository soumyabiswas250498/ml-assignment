# app.py
import streamlit as st
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# --- PAGE CONFIG ---
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
st.title("📞 Telco Customer Churn Prediction")
st.markdown("**BITS Pilani - Assignment 2** | Model: SMOTE Enhanced")

# --- 1. LOAD ARTIFACTS ---
MODEL_PATH = "model/outputModel"
if not os.path.exists(MODEL_PATH):
    st.error("Model folder not found. Please run the notebook first.")
    st.stop()


@st.cache_resource
def load_resources():
    try:
        scaler = pickle.load(open(f"{MODEL_PATH}/scaler.pkl", "rb"))
        feature_names = pickle.load(open(f"{MODEL_PATH}/features.pkl", "rb"))
        return scaler, feature_names
    except FileNotFoundError:
        return None, None


scaler, feature_names = load_resources()

if not scaler:
    st.error(
        "Artifacts missing. Run your training notebook to generate 'scaler.pkl' and 'features.pkl'."
    )
    st.stop()

# --- 2. SIDEBAR INPUTS ---
st.sidebar.header("Customer Profile")


def user_input_features():
    # Helper function to match your training columns
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.sidebar.number_input(
        "Total Charges ($)", 0.0, 10000.0, tenure * monthly_charges
    )

    # Categorical Selections
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

    # Create DataFrame
    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        # Default values for columns we didn't ask for (to keep UI simple)
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

# --- 3. PREDICTION ---
st.subheader("Prediction Interface")
model_list = [
    f
    for f in os.listdir(MODEL_PATH)
    if f.endswith(".pkl") and f not in ["scaler.pkl", "features.pkl"]
]
selected_model = st.selectbox("Select Model", model_list)

if st.button("Predict Churn"):
    # Preprocess
    input_encoded = pd.get_dummies(input_df)
    # Align columns to match training data exactly
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)
    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Load Model
    model = pickle.load(open(f"{MODEL_PATH}/{selected_model}", "rb"))

    # Predict
    pred = model.predict(input_scaled)[0]
    prob = (
        model.predict_proba(input_scaled)[0][1]
        if hasattr(model, "predict_proba")
        else 0
    )

    # Display
    if pred == 1:
        st.error(f"⚠️ High Churn Risk (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Safe Customer (Probability: {prob:.2%})")

# --- 4. BATCH UPLOAD (Assignment Req) ---
st.divider()
st.subheader("Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    test_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", test_data.head(3))

    # Preprocess & Predict
    if st.button("Run Batch Prediction"):
        # Basic Cleaning
        if "customerID" in test_data.columns:
            test_data = test_data.drop("customerID", axis=1)
        test_data["TotalCharges"] = pd.to_numeric(
            test_data["TotalCharges"], errors="coerce"
        ).fillna(0)

        # Prepare X and y
        if "Churn" in test_data.columns:
            y_true = test_data["Churn"].map({"Yes": 1, "No": 0})
            X_test = test_data.drop("Churn", axis=1)
        else:
            y_true = None
            X_test = test_data

        # Transform
        X_test_encoded = pd.get_dummies(X_test)
        X_test_encoded = X_test_encoded.reindex(columns=feature_names, fill_value=0)
        X_test_scaled = scaler.transform(X_test_encoded)

        # Predict
        model = pickle.load(open(f"{MODEL_PATH}/{selected_model}", "rb"))
        y_pred = model.predict(X_test_scaled)

        st.success("Predictions generated!")
        test_data["Prediction"] = y_pred
        st.write(test_data[["Prediction"]].head())

        if y_true is not None:
            acc = accuracy_score(y_true, y_pred)
            st.metric("Batch Accuracy", f"{acc:.2%}")
