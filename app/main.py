from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pandas as pd
from pathlib import Path

app = FastAPI()

# Load preprocessing pipeline and model
preprocessor = load(Path("../models/preprocessor.joblib"))
model = load(Path("../models/logistic_regression_model.joblib"))

# Define request schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
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
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(data: CustomerData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Preprocess features
    X_processed = preprocessor.transform(df)
    
    # Predict probability of churn
    prob = model.predict_proba(X_processed)[:, 1][0]
    
    return {"churn_probability": prob}
