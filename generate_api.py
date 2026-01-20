import joblib
import json
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
OUTPUT_PATH = Path("main.py")

# Load metadata
feature_names = joblib.load(ARTIFACTS_DIR / "feature_names.pkl")

with open(ARTIFACTS_DIR / "best_model.json") as f:
    metadata = json.load(f)

model_name = metadata["model_name"]
threshold = metadata.get("threshold", 0.5)

# Generate Pydantic schema fields
schema_fields = "\n".join([f"    {f}: str | int | float" for f in feature_names])

# FastAPI template
template = f'''
from fastapi import FastAPI
from pydantic import BaseModel
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
THRESHOLD = metadata.get("threshold", {threshold})

app = FastAPI(
    title="Auto-generated Bank Marketing API",
    version="1.0"
)

class Client(BaseModel):
{schema_fields}

class BatchClients(BaseModel):
    clients: List[Client]

@app.get("/health")
def health():
    return {{"status": "ok", "model": MODEL_NAME}}

@app.post("/predict")
def predict(client: Client):
    df = pd.DataFrame([client.dict()])
    X = preprocessor.transform(df)
    proba = float(model.predict_proba(X)[:, 1][0])
    return {{
        "model": MODEL_NAME,
        "probability": proba,
        "prediction": int(proba >= THRESHOLD)
    }}

@app.post("/predict_batch")
def predict_batch(batch: BatchClients):
    df = pd.DataFrame([c.dict() for c in batch.clients])
    X = preprocessor.transform(df)
    probas = model.predict_proba(X)[:, 1].tolist()
    return {{
        "model": MODEL_NAME,
        "results": [
            {{"probability": float(p), "prediction": int(p >= THRESHOLD)}} for p in probas
        ]
    }}
'''

OUTPUT_PATH.write_text(template.strip())
print("✅ main.py generated automatically")
