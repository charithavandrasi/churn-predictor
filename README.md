# 📊 Customer Churn Prediction App

A machine learning web application that predicts whether a telecom customer is likely to churn (leave the service), built using Python and deployed with Streamlit.

🌐 **Live Demo:** [https://churn-predictor-1605130281.streamlit.app](https://churn-predictor-1605130281.streamlit.app)

---

## 📌 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Losing a customer is far more expensive than retaining one. This app helps businesses identify at-risk customers early so they can take action before it's too late.

---

## 🎯 What This App Does

- Takes customer details as input (contract type, monthly charges, tenure, etc.)
- Predicts whether the customer will churn or stay
- Shows the churn probability as a percentage
- Displays the top factors that drive churn

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas & NumPy | Data cleaning and manipulation |
| Scikit-learn | Machine learning model |
| Matplotlib & Seaborn | Data visualization |
| Streamlit | Web app framework |
| GitHub | Version control |

---

## 📂 Project Structure

```
churn-predictor/
│
├── data/
│   └── telco_churn.csv          # IBM Telco Customer Churn dataset
│
├── churn_model.ipynb            # EDA, preprocessing and model training
├── app.py                       # Streamlit web application
├── churn_model.pkl              # Saved Logistic Regression model
├── scaler.pkl                   # Saved StandardScaler
├── feature_columns.pkl          # Saved feature column names
└── requirements.txt             # Python dependencies
```

---

## 📊 Dataset

- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target:** Churn (Yes / No)
- **Churn Rate:** ~27% of customers churned

---

## 🤖 Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **82.11%** ✅ |
| Random Forest | 79.35% |

Logistic Regression was selected as the final model due to its higher accuracy and better recall on churned customers.

---

## 🔍 Key Insights

The top 3 factors driving customer churn are:

1. **Total Charges** — Customers who have paid more overall tend to churn more
2. **Monthly Charges** — Higher monthly bills increase churn risk
3. **Tenure** — Newer customers churn more than long-term loyal customers

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/charithavandrasi/churn-predictor.git
cd churn-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open your browser at** `http://localhost:8501`

---

## 📈 Future Improvements

- [ ] Add AI chatbot using Gemini API to answer questions about the data
- [ ] Add SHAP values for better model explainability
- [ ] Try XGBoost and compare performance
- [ ] Add batch prediction feature (upload a CSV of customers)

---

## 👩‍💻 About

Built by **Charitha Vandrasi** as part of a 3-project data science portfolio.

- 🔗 [LinkedIn](https://www.linkedin.com/in/cvandrasi/)
- 💻 [GitHub](https://github.com/charithavandrasi/churn-predictor)

---

⭐ If you found this project helpful, please give it a star!