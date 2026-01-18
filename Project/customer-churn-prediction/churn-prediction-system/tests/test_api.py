"""
Unit tests for the Customer Churn Prediction API
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_predict_with_valid_data():
    """Test prediction endpoint with valid data"""
    payload = {
        "tenure": 24,
        "MonthlyCharges": 65.5,
        "TotalCharges": 1570.5,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "churn_probability" in response.json()


def test_predict_with_invalid_data():
    """Test prediction endpoint with invalid data"""
    payload = {
        "tenure": "invalid"  # Should be int
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error
