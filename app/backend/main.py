from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import psutil
import os
import numpy as np
from joblib import load
import pandas as pd
from pathlib import Path

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing; in production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load preprocessing pipeline and model
preprocessor = load(Path("../../models/preprocessor.joblib"))
model = load(Path("../../models/logistic_regression_model.joblib"))

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
    # Record memory at start
    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / (1024 * 1024)  # in MB

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Preprocess features
    X_processed = preprocessor.transform(df)
    
    # Predict probability of churn
    prob = model.predict_proba(X_processed)[:, 1][0]

    # Record memory at end
    mem_end = process.memory_info().rss / (1024 * 1024)  # in MB

    print(f"[MEMORY] Before processing: {mem_start:.2f} MB")
    print(f"[MEMORY] After processing: {mem_end:.2f} MB")
    print(f"[MEMORY] Used for this request: {mem_end - mem_start:.2f} MB")

    return {"churn_probability": prob}

