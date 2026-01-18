import joblib
import pandas as pd
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "model_artifacts" / "churn_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "model_artifacts" / "preprocessor.pkl"

# Load model and preprocessor (with graceful fallback for demo)
model = None
preprocessor = None

try:
    if MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        ARTIFACTS_LOADED = True
    else:
        ARTIFACTS_LOADED = False
except Exception as e:
    print(f"Warning: Could not load artifacts: {e}")
    ARTIFACTS_LOADED = False

def predict_churn(data):
    """
    Predict churn probability for a customer.
    
    If model artifacts are not available, returns a demo response.
    """
    if not ARTIFACTS_LOADED:
        # Demo mode: return sample prediction
        return {
            "status": "demo",
            "message": "Model artifacts not yet trained. Run build_features.py and train_model.py first.",
            "churn_probability": 0.35,
            "risk_level": "MEDIUM",
            "note": "This is a demo response. Real predictions require model training."
        }
    
    try:
        df = pd.DataFrame([data])
        processed = preprocessor.transform(df)
        prob = model.predict_proba(processed)[0][1]

        if prob > 0.7:
            risk = "HIGH"
        elif prob > 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "churn_probability": round(prob, 3),
            "risk_level": risk
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Prediction failed. Check input data format."
        }
