import requests
import json

# ⚡ Replace this with your actual Render URL
BASE_URL = "https://bank-deposit-subscription-api.onrender.com"

# -----------------------------
# 1️⃣ Health check
# -----------------------------
try:
    response = requests.get(f"{BASE_URL}/health")
    print("Health check:", response.json())
except Exception as e:
    print("Health check failed:", e)

# -----------------------------
# 2️⃣ Test single transaction
# -----------------------------
try:
    with open("transaction.json", "r") as f:
        transaction_data = json.load(f)

    response = requests.post(f"{BASE_URL}/predict", json=transaction_data)
    print("\nSingle transaction prediction:")
    print(response.json())
except Exception as e:
    print("Single transaction test failed:", e)

# -----------------------------
# 3️⃣ Test batch transactions
# -----------------------------
try:
    with open("batch_transactions.json", "r") as f:
        batch_data = json.load(f)

    response = requests.post(f"{BASE_URL}/predict_batch", json=batch_data)
    print("\nBatch transactions predictions:")
    print(response.json())
except Exception as e:
    print("Batch transaction test failed:", e)
