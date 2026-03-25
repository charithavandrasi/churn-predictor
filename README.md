# 📊 Customer Churn Prediction App — AI Powered

An end-to-end machine learning web application that predicts whether a telecom customer is likely to churn, complete with an AI-powered chatbot assistant for business insights.
🌐 **Live Demo:** [https://churn-predictor-1605130281.streamlit.app](https://churn-predictor-1605130281.streamlit.app)

💻 **GitHub:** [https://github.com/charithavandrasi/churn-predictor](https://github.com/charithavandrasi/churn-predictor)

---

## 📌 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Losing a customer is far more expensive than retaining one. This app helps businesses identify at-risk customers early so they can take action before it's too late.

---

## 🎯 What This App Does

### Tab 1 — 📊 Churn Predictor
- Takes 19 customer details as input (contract type, monthly charges, tenure, etc.)
- Predicts whether the customer will churn or stay
- Shows the churn probability as a percentage with a live progress bar
- Displays the top 10 factors driving churn using a feature importance chart

### Tab 2 — 🤖 AI Assistant
- Powered by **LLaMA 3.3 70B** via Groq API
- Answers business questions about the dataset in plain English
- Includes suggested questions to get started instantly
- Full chat history maintained during the session

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Pandas & NumPy | Data cleaning and manipulation |
| Scikit-learn | Machine learning model training |
| Matplotlib & Seaborn | Data visualization |
| Streamlit | Web app framework |
| Groq + LLaMA 3.3 70B | AI chatbot assistant |
| GitHub | Version control |
| Streamlit Cloud | Free deployment |

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
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 📊 Dataset

- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target:** Churn (Yes / No)
- **Churn Rate:** ~26.5% of customers churned

---

## 🤖 Model Performance

| Model | Accuracy | Churn Recall |
|-------|----------|-------------|
| Logistic Regression | **82.11%** ✅ | 60% |
| Random Forest | 79.35% | 47% |

Logistic Regression was selected as the final model due to its higher overall accuracy and better recall on churned customers — missing a churner is more costly than a false alarm.

---

## 🔍 Key Business Insights

The top factors driving customer churn are:

1. **Total Charges** — Customers who have paid more overall tend to churn more, suggesting value perception issues
2. **Monthly Charges** — Higher monthly bills significantly increase churn risk
3. **Tenure** — Newer customers churn more; loyalty grows with time
4. **Contract Type** — Month-to-month customers churn far more than yearly contract customers
5. **Internet Service** — Fiber optic customers churn more than DSL customers
6. **Online Security** — Customers without online security are more likely to leave

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

**5. For the AI chatbot** — get a free Groq API key at 👉 https://console.groq.com

---

## 🧠 How the AI Chatbot Works

The AI Assistant tab uses:
- **Groq API** — ultra-fast inference engine
- **LLaMA 3.3 70B** — Meta's powerful open source language model
- A custom system prompt that gives the AI full context about our dataset, model performance and key insights
- Streamlit session state to maintain full chat history

Example questions you can ask:
- *"What type of customers churn the most?"*
- *"How can the business reduce churn rate?"*
- *"Why does tenure affect churn?"*
- *"What does 82% accuracy mean in business terms?"*

---

## 📈 Future Improvements

- [ ] Add SHAP values for deeper model explainability
- [ ] Try XGBoost and compare with Logistic Regression
- [ ] Add batch prediction — upload a CSV of customers
- [ ] Connect chatbot to live prediction results
- [ ] Add customer segmentation analysis

---

## 🗂️ Project Workflow

```
Raw Data → EDA → Cleaning → Preprocessing → Model Training →
Model Evaluation → Streamlit App → AI Chatbot → Deployment
```

---

## 👩‍💻 About

Built by **Charitha Vandrasi** as Project 2 of a 3-project data science portfolio.

- 🔗 [LinkedIn](https://www.linkedin.com/in/cvandrasi/)
- 💻 [GitHub](https://github.com/charithavandrasi)

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit==1.40.0
groq
```

---

⭐ If you found this project helpful, please give it a star!