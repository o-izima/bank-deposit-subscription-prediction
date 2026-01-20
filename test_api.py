import requests
import json

BASE_URL = "http://localhost:8000"

# -----------------------------
# Health check
# -----------------------------
health = requests.get(f"{BASE_URL}/health")
print("Health check:", health.json())

# -----------------------------
# Single prediction
# -----------------------------
with open("transaction.json", "r") as f:
    transaction_data = json.load(f)

response = requests.post(
    f"{BASE_URL}/predict",
    json=transaction_data
)

print("\nSingle transaction prediction:")
print(response.json())

# -----------------------------
# Batch prediction
# -----------------------------
with open("batch_transactions.json", "r") as f:
    batch_data = json.load(f)

response = requests.post(
    f"{BASE_URL}/predict_batch",
    json=batch_data
)

print("\nBatch transactions predictions:")
print(response.json())
