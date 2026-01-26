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

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

# --- CONFIGURATION ---
# Ensure this matches your folder name
MODEL_FOLDER = "model"

st.title("📞 Telco Customer Churn Prediction")
st.markdown(
    """
**BITS Pilani - Assignment 2**
"""
)


# --- 1. LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_FOLDER):
        st.error(f"⚠️ Error: Folder '{MODEL_FOLDER}' not found.")
        return None, None
    try:
        scaler = pickle.load(open(os.path.join(MODEL_FOLDER, "scaler.pkl"), "rb"))
        features = pickle.load(open(os.path.join(MODEL_FOLDER, "features.pkl"), "rb"))
        return scaler, features
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None


scaler, feature_names = load_artifacts()

if not scaler:
    st.stop()

# --- 2. SIDEBAR: MODEL SELECTION [Req: b] ---
st.sidebar.header("⚙️ Model Configuration")
available_models = [
    f
    for f in os.listdir(MODEL_FOLDER)
    if f.endswith(".pkl") and f not in ["scaler.pkl", "features.pkl"]
]
selected_model_file = st.sidebar.selectbox("Select Model", available_models)

# --- 3. SINGLE PREDICTION TAB ---
tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Batch Prediction (Assignment)"])

with tab1:
    st.subheader("Single Customer Prediction")

    # Input Form
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.number_input("Tenure (Months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col3:
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        total_charges = st.number_input(
            "Total Charges ($)", 0.0, 10000.0, tenure * monthly_charges
        )

    if st.button("Predict Single Customer"):
        # Prepare Data
        input_data = pd.DataFrame(
            {
                "tenure": [tenure],
                "MonthlyCharges": [monthly_charges],
                "TotalCharges": [total_charges],
                "Contract": [contract],
                "InternetService": [internet],
                "PaymentMethod": [payment],
                # Defaults
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
                "SeniorCitizen": 0,
            }
        )

        # Process
        input_encoded = pd.get_dummies(input_data).reindex(
            columns=feature_names, fill_value=0
        )
        input_scaled = scaler.transform(input_encoded)

        # Predict
        model = pickle.load(open(os.path.join(MODEL_FOLDER, selected_model_file), "rb"))
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Churn Risk: {prob:.1%}")
        else:
            st.success(f"✅ Safe: {prob:.1%}")

# --- 4. BATCH PREDICTION TAB [Req: a, c, d] ---
with tab2:
    st.subheader("Batch Prediction with Evaluation")

    # [Req: a] Dataset Upload Option
    uploaded_file = st.file_uploader("Upload CSV (Test Data)", type=["csv"])

    if uploaded_file and selected_model_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", df.head())

        if st.button("Run Batch Prediction"):
            # 1. Preprocessing
            if "customerID" in df.columns:
                df_clean = df.drop("customerID", axis=1)
            else:
                df_clean = df.copy()

            # Handle TotalCharges
            if "TotalCharges" in df_clean.columns:
                df_clean["TotalCharges"] = pd.to_numeric(
                    df_clean["TotalCharges"], errors="coerce"
                ).fillna(0)

            # Separate Target if exists
            y_true = None
            if "Churn" in df_clean.columns:
                # Map Yes/No to 1/0
                y_true = df_clean["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
                X = df_clean.drop("Churn", axis=1)
            else:
                X = df_clean

            # Encode & Scale
            X_encoded = pd.get_dummies(X).reindex(columns=feature_names, fill_value=0)
            X_scaled = scaler.transform(X_encoded)

            # Predict
            model = pickle.load(
                open(os.path.join(MODEL_FOLDER, selected_model_file), "rb")
            )
            y_pred = model.predict(X_scaled)

            # Display Results
            df["Prediction"] = y_pred
            df["Prediction_Label"] = df["Prediction"].map({1: "Yes", 0: "No"})
            st.dataframe(df[["Prediction_Label", "Prediction"]])

            # [Req: c & d] EVALUATION METRICS (Only if Ground Truth exists)
            if y_true is not None:
                st.divider()
                st.subheader("📊 Evaluation Report")

                # [Req: c] Metrics Display
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
                m2.metric("Precision", f"{precision_score(y_true, y_pred):.2f}")
                m3.metric("Recall", f"{recall_score(y_true, y_pred):.2f}")
                m4.metric("F1 Score", f"{f1_score(y_true, y_pred):.2f}")

                # [Req: d] Confusion Matrix & Classification Report
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
                    report_dict = classification_report(
                        y_true, y_pred, output_dict=True
                    )
                    st.dataframe(pd.DataFrame(report_dict).transpose())
            else:
                st.info(
                    "Upload a dataset with a 'Churn' column to see Evaluation Metrics."
                )
