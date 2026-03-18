import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load model files ──────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_model()

# ── Header ────────────────────────────────────────────────────
st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details in the sidebar to predict whether they will churn.")
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", 
                    ["Fiber optic", "DSL", "No"])
online_security = st.sidebar.selectbox("Online Security", 
                    ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", 
                    ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", 
                    ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", 
                    ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", 
                    ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", 
                    ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract Type", 
                    ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox("Payment Method", [
                    "Electronic check", "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 
                    0.0, 120.0, 65.0)
total_charges = st.sidebar.slider("Total Charges ($)", 
                    0.0, 9000.0, 1500.0)

# ── Build input dataframe ─────────────────────────────────────
def build_input():
    data = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'MultipleLines': 1 if multiple_lines == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless == 'Yes' else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
        'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
        'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
        'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
        'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
    }
    input_df = pd.DataFrame([data])
    # Make sure columns match training data exactly
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return input_df

# ── Prediction ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prediction Result")
    if st.button("🔍 Predict Churn", use_container_width=True):
        input_df = build_input()
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Churn Risk: {probability * 100:.1f}% probability")
            st.markdown("This customer is **likely to churn**. Consider offering a discount or upgrade.")
        else:
            st.success(f"✅ Low Churn Risk: {probability * 100:.1f}% probability")
            st.markdown("This customer is **likely to stay**. Keep up the good service!")

        # Probability bar
        st.markdown("**Churn Probability:**")
        st.progress(float(probability))

with col2:
    st.subheader("Key Churn Drivers")
    # Feature importance chart
    with open('feature_columns.pkl', 'rb') as f:
        feat_cols = pickle.load(f)

    importance_df = pd.DataFrame({
        'Feature': feat_cols,
        'Importance': model.coef_[0]
    }).sort_values('Importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=importance_df, x='Importance', 
                y='Feature', hue='Feature',
                legend=False, palette='viridis', ax=ax)
    ax.set_title('Top Factors Influencing Churn')
    st.pyplot(fig)