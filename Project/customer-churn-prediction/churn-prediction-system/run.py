#!/usr/bin/env python
"""Run the FastAPI application"""
import sys
from pathlib import Path
import uvicorn

# Add the churn-prediction-system directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
