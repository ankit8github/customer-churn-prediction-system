
# Customer Churn Prediction System

**Production-ready end-to-end ML system** for predicting customer churn using historical usage and account data. Built with real-world ML engineering practices and deployed via a FastAPI-based REST API.

**Tech:** Python, Pandas, Scikit-learn, XGBoost, FastAPI, Streamlit
**Status:** v1.0.0 | Python 3.8+


## 🔍 Overview

Customer churn is a key business challenge in SaaS, Telecom, and FinTech.
This project predicts churn probability and categorizes customers into actionable risk levels to enable proactive retention strategies.

**Key Features**

* End-to-end ML pipeline (EDA → Training → Deployment)
* Probability-based churn prediction with risk levels
* Consistent preprocessing for training & inference
* Scalable, stateless FastAPI service

## 🏗️ Architecture

```
Raw Data → Feature Engineering → Model Training
        → Saved Artifacts → FastAPI Inference → Deployment
```


## 📊 Data & Insights

* Dataset: Telco Customer Churn (1000+ records)
* Target: `Churn` (Yes / No)

**Key Insights**

* High churn for customers with <6 months tenure
* Month-to-month contracts show >50% churn
* Lack of tech support increases churn risk
* Higher monthly charges correlate with churn

## 🤖 Models & Performance

**Models Tested**

* Logistic Regression
* Random Forest
* XGBoost (final model)

**Why XGBoost?**

* Best ROC-AUC & F1-score
* Handles class imbalance well
* Low-latency inference

**Test Performance**

* ROC-AUC: **0.84**
* F1-score: **0.68**
* Accuracy: **80.5%**

## 🚀 API & Inference

**Framework:** FastAPI
**Flow:**

1. JSON input validation
2. Saved preprocessing pipeline
3. Churn probability prediction
4. Risk classification

**Risk Levels**

* LOW: < 0.40
* MEDIUM: 0.40–0.70
* HIGH: > 0.70

## ⚙️ Tech Stack

* **ML:** Scikit-learn, XGBoost
* **Data:** Pandas, NumPy
* **API:** FastAPI, Uvicorn
* **Deployment:** Streamlit
* **Cloud & Tools:** Git, GitHub, Azure

## ▶️ Run Locally

```bash
git clone <repo-url>
cd churn-prediction-system
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**API Docs:** `http://127.0.0.1:8000/docs`

## 📌 Key Learnings

* End-to-end ML system design
* Handling class imbalance
* Production-ready inference pipelines
* API-first ML deployment

## 🔮 Future Work

* Model monitoring & drift detection
* Automated retraining
* SHAP explainability
* Authentication & rate limiting

## 👤 Author

**Ankit Kashyap**
📧 [ankit.kashyap0221@gmail.com](mailto:ankit.kashyap0221@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/ankitkashyap01)
💻 [GitHub](https://github.com/ankit8github)

