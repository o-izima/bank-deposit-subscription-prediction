from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import pandas as pd
import joblib
import json

# Load artifacts
preprocessor = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("models/best_model.pkl")

with open("artifacts/best_model.json") as f:
    metadata = json.load(f)

MODEL_NAME = metadata["model_name"]
THRESHOLD = metadata.get("threshold", 0.5)

app = FastAPI(
    title="Auto-generated Bank Marketing API",
    version="1.0"
)

# -----------------------------
# Input Schemas
# -----------------------------
class Client(BaseModel):
    age: int | float
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: int | float
    campaign: int | float
    pdays: int | float
    previous: int | float
    poutcome: str

    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(..., alias="nr.employed")

    model_config = ConfigDict(populate_by_name=True)


class BatchClients(BaseModel):
    clients: List[Client]

# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def root():
    return {"message": "Bank Marketing Subscription Prediction API is live!"}
    
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict")
def predict(client: Client):
    df = pd.DataFrame([client.dict(by_alias=True)])
    X = preprocessor.transform(df)
    proba = float(model.predict_proba(X)[:, 1][0])

    return {
        "model": MODEL_NAME,
        "probability": proba,
        "prediction": int(proba >= THRESHOLD)
    }


@app.post("/predict_batch")
def predict_batch(batch: BatchClients):
    df = pd.DataFrame([c.dict(by_alias=True) for c in batch.clients])
    X = preprocessor.transform(df)
    probas = model.predict_proba(X)[:, 1].tolist()

    return {
        "model": MODEL_NAME,
        "results": [
            {"probability": float(p), "prediction": int(p >= THRESHOLD)}
            for p in probas
        ]
    }
