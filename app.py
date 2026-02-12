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

# Let's set up the page configuration first
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
st.title("📞 Telco Customer Churn Prediction")
st.markdown("**BITS Pilani - Assignment 2** | Model: SMOTE Enhanced")

# --- Loading the necessary files ---

MODEL_PATH = "model/outputModel"

if not os.path.exists(MODEL_PATH):
    st.error(f"Folder '{MODEL_PATH}' not found. Please run the notebook first.")
    st.stop()


@st.cache_resource
def load_resources():
    # Try to load the pickled files. If they aren't there, we'll handle it gracefully.
    try:
        scaler = pickle.load(open(f"{MODEL_PATH}/scaler.pkl", "rb"))
        feature_names = pickle.load(open(f"{MODEL_PATH}/features.pkl", "rb"))
        return scaler, feature_names
    except FileNotFoundError:
        return None, None


def preprocess_data(df, scaler, feature_names):
    # This function ensures the input data looks exactly like what the model expects.
    # It handles one-hot encoding and scaling.
    df_encoded = pd.get_dummies(df)
    # Reindex ensures we have all the columns the model was trained on, in the right order.
    # Missing columns are filled with 0.
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    return scaler.transform(df_encoded)


@st.cache_resource
def load_model(path):
    # We cache the model so we don't have to reload it from disk every time the user interacts with the app.
    return pickle.load(open(path, "rb"))


scaler, feature_names = load_resources()

if not scaler:
    st.error(
        "Artifacts missing. Run your training notebook to generate 'scaler.pkl' and 'features.pkl'."
    )
    st.stop()

# --- User Input Section ---
st.sidebar.header("Customer Profile")


def user_input_features():
    # We'll use a helper function to gather all the inputs from the sidebar
    # and pack them into a DataFrame.
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
    # We'll estimate total charges based on tenure and monthly charges for simplicity
    total_charges = st.sidebar.number_input(
        "Total Charges ($)", 0.0, 10000.0, tenure * monthly_charges
    )

    # Dropdowns for categorical features
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

    # Construct the DataFrame
    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        # These features are hidden from the UI and set to default values
        # to simplify the interface for this demo.
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

# --- Prediction Logic ---
st.subheader("Prediction Interface")
# List all available models in the directory
model_list = [
    f
    for f in os.listdir(MODEL_PATH)
    if f.endswith(".pkl") and f not in ["scaler.pkl", "features.pkl"]
]
selected_model = st.selectbox("Select Model", model_list)

if st.button("Predict Churn"):
    # First, get the data ready for the model
    input_scaled = preprocess_data(input_df, scaler, feature_names)

    # Load the chosen model
    model = load_model(f"{MODEL_PATH}/{selected_model}")

    # Make the prediction
    pred = model.predict(input_scaled)[0]
    prob = (
        model.predict_proba(input_scaled)[0][1]
        if hasattr(model, "predict_proba")
        else 0
    )

    # Show the result to the user
    if pred == 1:
        st.error(f"⚠️ High Churn Risk (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Safe Customer (Probability: {prob:.2%})")

# --- Batch Prediction Section ---
st.divider()
st.subheader("Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file:
    test_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", test_data.head(3))

    if st.button("Run Batch Prediction"):
        # Do some basic cleanup on the uploaded data
        if "customerID" in test_data.columns:
            test_data = test_data.drop("customerID", axis=1)

        # Make sure TotalCharges is numeric
        if "TotalCharges" in test_data.columns:
            test_data["TotalCharges"] = pd.to_numeric(
                test_data["TotalCharges"], errors="coerce"
            ).fillna(0)

        # Separate features and target if 'Churn' exists
        if "Churn" in test_data.columns:
            # Convert Yes/No to binary 1/0
            y_true = test_data["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
            X_test = test_data.drop("Churn", axis=1)
        else:
            y_true = None
            X_test = test_data

        # Preprocess the batch data
        X_test_scaled = preprocess_data(X_test, scaler, feature_names)

        # Run predictions
        model = load_model(f"{MODEL_PATH}/{selected_model}")
        y_pred = model.predict(X_test_scaled)

        st.success("Predictions generated!")
        test_data["Prediction"] = y_pred
        st.dataframe(test_data[["Prediction"]].head())

        # --- Evaluation Metrics ---
        # If we have the actual labels, we can show how well the model performed.
        if y_true is not None:
            st.divider()
            st.subheader("📊 Model Performance Report")

            # Display key metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
            m2.metric("Precision", f"{precision_score(y_true, y_pred):.2%}")
            m3.metric("Recall", f"{recall_score(y_true, y_pred):.2%}")
            m4.metric("F1 Score", f"{f1_score(y_true, y_pred):.2%}")

            # Show detailed reports
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
