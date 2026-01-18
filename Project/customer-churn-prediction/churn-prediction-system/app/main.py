from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.schemas import ChurnRequest, ChurnResponse
from app.predict import predict_churn, ARTIFACTS_LOADED
import json

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability using ML models",
    version="1.0.0"
)

# Global exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed error messages"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"][1:]),
            "type": error["type"],
            "message": error["msg"]
        })
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": errors,
            "hint": "Check the /docs endpoint for schema and example data"
        }
    )

@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "churn-prediction-api",
        "model_loaded": ARTIFACTS_LOADED
    }

@app.get("/", tags=["Info"])
def root():
    """Root endpoint with API info"""
    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        },
        "model_status": "Ready" if ARTIFACTS_LOADED else "Training Required"
    }

@app.post("/predict", response_model=ChurnResponse, tags=["Prediction"])
def predict(request: ChurnRequest):
    """
    Predict customer churn probability
    
    **Input**: ChurnRequest with customer features
    
    **Output**: 
    - churn_probability: Float (0-1)
    - risk_level: LOW | MEDIUM | HIGH
    """
    try:
        result = predict_churn(request.dict())
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "message": "Prediction failed. Check input data format."
            }
        )
