import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

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

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Churn Predictor", "🤖 AI Assistant"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — CHURN PREDICTOR
# ══════════════════════════════════════════════════════════════
with tab1:
    st.title("📊 Customer Churn Predictor")
    st.markdown("Enter customer details in the sidebar to predict whether they will churn.")
    st.divider()

    # ── Sidebar inputs ────────────────────────────────────────
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
    monthly_charges = st.sidebar.slider("Monthly Charges ($)", 0.0, 120.0, 65.0)
    total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

    # ── Build input dataframe ─────────────────────────────────
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
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        return input_df

    # ── Prediction ────────────────────────────────────────────
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

            st.markdown("**Churn Probability:**")
            st.progress(float(probability))

    with col2:
        st.subheader("Key Churn Drivers")
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': model.coef_[0]
        }).sort_values('Importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=importance_df, x='Importance',
                    y='Feature', hue='Feature',
                    legend=False, palette='viridis', ax=ax)
        ax.set_title('Top Factors Influencing Churn')
        st.pyplot(fig)

# ══════════════════════════════════════════════════════════════
# TAB 2 — AI ASSISTANT
# ══════════════════════════════════════════════════════════════
with tab2:
    st.title("🤖 AI Churn Analysis Assistant")
    st.markdown("Ask me anything about customer churn, the dataset, or what the business should do!")
    st.divider()

    # ── API Key input ─────────────────────────────────────────
    api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free API key at https://console.groq.com"
    )

    # ── Chat history ──────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Suggested questions (only show when no messages yet) ──
    if len(st.session_state.messages) == 0:
        st.markdown("**💡 Try asking:**")
        col1, col2 = st.columns(2)

        suggested = None
        with col1:
            if st.button("What type of customers churn the most?"):
                suggested = "What type of customers churn the most?"
            if st.button("How can the business reduce churn?"):
                suggested = "How can the business reduce churn?"
        with col2:
            if st.button("Why does tenure affect churn?"):
                suggested = "Why does tenure affect churn?"
            if st.button("What does the model accuracy of 82% mean?"):
                suggested = "What does the model accuracy of 82% mean?"

        if suggested:
            st.session_state.messages.append({"role": "user", "content": suggested})
            if api_key:
                try:
                    client = Groq(api_key=api_key)
                    context = """
                    You are an expert data scientist assistant for a Customer Churn Prediction project.
                    Dataset: IBM Telco Customer Churn, 7043 customers, 21 features.
                    Churn rate: 26.5%. Best model: Logistic Regression 82.11% accuracy.
                    Top churn factors: Total Charges, Monthly Charges, Tenure.
                    Key insights: Month-to-month contracts churn most. Fiber optic customers churn more. No online security = higher churn.
                    Answer clearly and in a business-friendly way.
                    """
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": context},
                            {"role": "user", "content": suggested}
                        ]
                    )
                    reply = response.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
            st.rerun()

    # ── Display all previous messages ─────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────
    st.divider()
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input("", placeholder="Ask a question about churn...", label_visibility="collapsed", key="chat_input")
    with col_btn:
        send = st.button("Send ➤", use_container_width=True)

    prompt = user_input if send and user_input else None
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not api_key:
            with st.chat_message("assistant"):
                st.warning("Please enter your Groq API key above to use the AI assistant.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = Groq(api_key=api_key)

                        context = """
                        You are an expert data scientist assistant for a Customer Churn Prediction project.
                        Here are the key facts about this project:
                        - Dataset: IBM Telco Customer Churn with 7,043 customers and 21 features
                        - Churn rate: 26.5% of customers churned
                        - Best model: Logistic Regression with 82.11% accuracy
                        - Top churn factors: Total Charges, Monthly Charges, Tenure
                        - Key insight: Month-to-month contract customers churn far more than yearly contract customers
                        - Key insight: Fiber optic internet customers churn more than DSL customers
                        - Key insight: Customers with no online security churn more
                        Answer questions clearly, concisely and in a business-friendly way.
                        """

                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": context},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        reply = response.choices[0].message.content
                        st.markdown(reply)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": reply
                        })

                    except Exception as e:
                        st.error(f"Error: {str(e)}. Please check your API key.")