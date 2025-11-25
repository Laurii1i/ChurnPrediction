# Churn Prediction Web App

This project is a **Customer Churn Prediction** application built with **FastAPI** for the backend and a lightweight **HTML/JavaScript frontend**. It predicts the probability that a customer will churn based on their account and usage information.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Backend](#backend)
- [Frontend](#frontend)
- [Usage](#usage)
- [Deployment](#deployment)
- [License](#license)

---

## Project Overview

Customer churn is when a customer stops using a company's product or service. Predicting churn allows businesses to proactively retain valuable customers.  

This project includes:
- A **trained Logistic Regression model** for churn prediction.
- A **FastAPI backend** exposing a `/predict` endpoint.
- A **frontend** that allows users to enter customer data and get churn probability.

---

## Dataset

The data is based on typical customer account and service information, with features including:

- `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- `tenure`, `PhoneService`, `MultipleLines`
- `InternetService`, `OnlineSecurity`, `OnlineBackup`
- `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- `Contract`, `PaperlessBilling`, `PaymentMethod`
- `MonthlyCharges`, `TotalCharges`

> The dataset contains a mix of categorical and numerical features. Categorical features are encoded for model input.

---

## Model

We use **Logistic Regression**, a classical and interpretable model for binary classification:

- Input features are **preprocessed using a pipeline** (e.g., one-hot encoding for categorical features and scaling for numerical features).  
- Output is a **churn probability** between 0 and 1.  

This model is lightweight, fast, and suitable for deployment on small server instances.

---

## Backend

The backend is implemented using **FastAPI**:

- Provides a `/predict` **POST endpoint**.
- Uses **Pydantic models** to validate incoming data.
- Loads the **preprocessing pipeline** and **trained Logistic Regression model** using `joblib`.
- Applies **CORS middleware** to allow frontend requests.
- Optionally serves the frontend via `StaticFiles`.

**Key files**:
- `backend/main.py` – FastAPI app, endpoint, and model loading.
- `models/` – Contains the preprocessor and trained model joblib files.
- `requirements.txt` – Dependencies for deployment (FastAPI, Pydantic, Pandas, Numpy, Gunicorn, Uvicorn, Scikit-learn, Joblib).

---

## Frontend

The frontend is a simple **HTML + JavaScript** interface:

- Collects all customer fields in a visually appealing form.
- Sends a JSON payload to the backend `/predict` endpoint.
- Displays the churn probability dynamically on the page.
- Uses two-column layout for better readability and user experience.

---

## Usage

### Running locally

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Mac/Linux
