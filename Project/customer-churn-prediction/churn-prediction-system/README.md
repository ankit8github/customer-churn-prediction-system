
# Customer Churn Prediction System

**Status:** Production Ready
**Version:** 1.0.0
**Python:** 3.8+

An end-to-end machine learning system that predicts customer churn using historical usage and account data. The project follows real-world ML engineering practices, including data preprocessing, feature engineering, model training, evaluation, and deployment via a production-ready REST API.

---

## 1. Project Overview

Customer churn is a critical business challenge for SaaS, Telecom, FinTech, and subscription-based organizations. This system enables early identification of high-risk customers, allowing businesses to take proactive retention actions such as targeted outreach, incentives, or support interventions.

**Objective:**
Predict the probability of customer churn using supervised machine learning and expose predictions through a scalable REST API.

**Key Outcomes:**

* Early detection of at-risk customers
* Probability-based churn prediction with actionable risk levels
* Consistent preprocessing across training and inference
* Production-ready FastAPI-based deployment

---

## 2. System Architecture

```
Raw Data
   ↓
Data Cleaning & Feature Engineering
   ↓
Model Training & Evaluation
   ↓
Model & Preprocessor Artifacts
   ↓
FastAPI Inference Service
   ↓
Production Deployment
```

**Architecture Highlights:**

* Clear separation between training and inference pipelines
* Reusable and versioned model artifacts
* Stateless API design suitable for horizontal scaling

---

## 3. Dataset

**Source:** Telco Customer Churn Dataset
**Records:** 1000+ customers
**Target Variable:** `Churn` (Yes / No)

**Feature Categories:**

* **Numerical:** Tenure, Monthly Charges, Total Charges, Senior Citizen
* **Categorical:** Gender, Partner, Dependents
* **Services:** Internet Service, Online Security, Backup, Tech Support, Streaming
* **Account Details:** Contract Type, Payment Method, Paperless Billing

**Data Paths:**

* Raw data: `data/raw/Telco-Customer-Churn.csv`
* Processed data: `data/processed/clean_churn_data.csv`

---

## 4. Exploratory Data Analysis (EDA)

Key insights derived from analysis:

1. Customers with tenure less than 6 months show significantly higher churn
2. Month-to-month contracts exhibit over 50% churn compared to long-term contracts
3. Absence of technical support increases churn likelihood by ~40%
4. Higher monthly charges correlate with elevated churn risk
5. Fiber optic internet users churn more frequently than DSL users

**Notebook:** `notebooks/data_understanding.ipynb`

---

## 5. Model Training & Evaluation

**Models Evaluated:**

* Logistic Regression (baseline, interpretable)
* Random Forest (ensemble-based, feature importance)
* XGBoost (gradient boosting, final production model)

**Evaluation Metrics:**

* Primary: ROC-AUC
* Secondary: F1-Score, Precision, Recall, Confusion Matrix

**Final Model Selection:** XGBoost
**Rationale:**

* Highest ROC-AUC and F1-score
* Low-latency inference
* Stable performance on imbalanced data
* Feature importance available for interpretability

**Training Script:** `src/models/train_model.py`
**Evaluation Summary:** `model_artifacts/model_comparison.csv`

---

## 6. Model Artifacts

Stored in `model_artifacts/` for reproducible inference:

| File                   | Description                        |
| ---------------------- | ---------------------------------- |
| `churn_model.pkl`      | Trained XGBoost model              |
| `preprocessor.pkl`     | Fitted preprocessing pipeline      |
| `feature_names.pkl`    | Feature validation reference       |
| `model_metadata.pkl`   | Training configuration and metrics |
| `model_comparison.csv` | Model performance comparison       |
| `train_test_data.pkl`  | Reproducible train-test split      |

---

## 7. Inference & API Design

**Framework:** FastAPI
**Inference Flow:**

1. JSON input validation via Pydantic
2. Consistent preprocessing using saved pipeline
3. Churn probability prediction
4. Risk categorization based on thresholds

**Risk Levels:**

* LOW: Probability < 0.40
* MEDIUM: 0.40 – 0.70
* HIGH: > 0.70

**API Modules:**
`app/main.py`, `app/predict.py`, `app/schemas.py`

---

## 8. Tech Stack

| Layer            | Technology            |
| ---------------- | --------------------- |
| Language         | Python 3.8+           |
| Data Processing  | Pandas, NumPy         |
| Machine Learning | scikit-learn, XGBoost |
| Serialization    | Joblib                |
| API Framework    | FastAPI, Uvicorn      |
| Configuration    | python-dotenv         |
| Testing          | Pytest                |
| Version Control  | Git, GitHub           |

---

## 9. Getting Started

### Prerequisites

* Python 3.8+
* pip
* Virtual environment (recommended)

### Installation

```bash
git clone <repository-url>
cd churn-prediction-system
```

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

```bash
pip install -r requirements.txt
```

### Train the Model (if artifacts are missing)

```bash
cd src/features
python build_features.py

cd ../models
python train_model.py
```

---

## 10. Running the API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**API Access:**

* Swagger UI: `http://127.0.0.1:8000/docs`
* ReDoc: `http://127.0.0.1:8000/redoc`
* Health Check: `http://127.0.0.1:8000/health`

---

## 11. Example Prediction

**Request:**

```json
{
  "tenure": 24,
  "MonthlyCharges": 65.5,
  "TotalCharges": 1570.5,
  "SeniorCitizen": 0,
  "gender": "Male",
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "InternetService": "Fiber optic",
  "Contract": "Two year",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
```

**Response:**

```json
{
  "churn_probability": 0.285,
  "risk_level": "LOW"
}
```

---

## 12. Performance Summary

**Model (Test Set):**

* ROC-AUC: 0.84
* F1-Score: 0.68
* Precision: 0.72
* Recall: 0.65
* Accuracy: 80.5%

**API:**

* Avg latency: ~50 ms
* Throughput: ~20 requests/sec (single worker)

---

## 13. Key Learnings

* End-to-end ML pipeline design
* Handling class imbalance in real datasets
* Consistent preprocessing for production inference
* API-first ML deployment
* Business-oriented metric selection

---

## 14. Future Enhancements

* Model monitoring and drift detection
* Automated retraining (CI/CD)
* SHAP-based explainability
* Authentication and rate limiting
* Feature store integration
* Cost-sensitive churn optimization

---

## 15. License

MIT License

---

## 16. Author & Contact

**Ankit Kashyap**
📧 [ankit.kashyap0221@gmail.com]
🔗 LinkedIn: [https://www.linkedin.com/in/ankitkashyap01]
💻 GitHub: [https://github.com/ankit8github]

---

**Last Updated:** January 2026
**Release:** v1.0.0 (Production Ready)
