from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class ChurnRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
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
    })
    
    # Numeric features
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int = 0
    
    # Categorical features
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

class ChurnResponse(BaseModel):
    churn_probability: Optional[float] = None
    risk_level: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    note: Optional[str] = None
