import json
import os
import requests

# =========================
# Environment config
# =========================
API_ENV = os.getenv("API_ENV", "local")

BASE_URL = (
    "https://bank-deposit-subscription-api.onrender.com"
    if API_ENV == "cloud"
    else "http://localhost:8000"
)

CLIENT_JSON = "sample_requests/client.json"
BATCH_JSON = "sample_requests/batch_clients.json"


# =========================
# Single prediction test
# =========================
def test_single_prediction():
    if not os.path.exists(CLIENT_JSON):
        raise FileNotFoundError(f"{CLIENT_JSON} not found. Check the path.")

    with open(CLIENT_JSON) as f:
        payload = json.load(f)

    response = requests.post(f"{BASE_URL}/predict", json=payload)
    response.raise_for_status()

    result = response.json()

    print("\n--- Single Client Prediction ---")
    print(result)

    if "predicted_class" in result:
        pred_class = result["predicted_class"]
    elif "prediction" in result:
        pred_class = result["prediction"]
    else:
        pred_class = result.get("class")

    proba = result.get("subscription_proba") or result.get("probability")

    print(f"Predicted Class: {pred_class}")
    print(f"Subscription Probability: {proba:.4f}")


# =========================
# Batch prediction test
# =========================
def test_batch_prediction():
    if not os.path.exists(BATCH_JSON):
        raise FileNotFoundError(f"{BATCH_JSON} not found. Check the path.")

    with open(BATCH_JSON) as f:
        payload = json.load(f)

    response = requests.post(f"{BASE_URL}/predict_batch", json=payload)
    response.raise_for_status()

    data = response.json()

    print("\n--- Batch Predictions ---")
    print(f"Model Used: {data.get('model')}")

    for i, r in enumerate(data.get("results", []), 1):
        print(
            f"{i}. Prediction: {r.get('prediction')}, "
            f"Probability: {r.get('probability'):.4f}"
        )


# =========================
# ENTRY POINT (CRITICAL)
# =========================
if __name__ == "__main__":
    print(f"Using API URL: {BASE_URL} (Environment: {API_ENV})")

    test_single_prediction()
    test_batch_prediction()
