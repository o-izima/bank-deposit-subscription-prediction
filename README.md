<a name="top"></a>
# Bank Deposit Subscription Prediction – Production-Ready ML Pipeline with FastAPI, Docker, and Render Deployment

This machine learning project predicts whether a client will subscribe to a bank deposit campaign using XGBoost, the best-performing model among all models trained. The project provides **Jupyter notebooks for EDA, preprocessing, and model training**, along with a **FastAPI-based REST API** deployed both locally via Docker **and in the cloud on Render**, enabling real-time subscription predictions.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Project Structure](#project-structure)  
- [Business Objective](#business-objective)  
- [Dataset](#dataset)  
- [Data Preparation](#data-preparation)  
- [Modeling Approach](#modeling-approach)  
- [Model Evaluation](#model-evaluation)  
- [Metrics Comparison](#metrics-comparison)  
- [Deployment Workflow](#deployment-workflow)  
- [Requirements](#requirements)  
- [Setup & Installation](#setup--installation)  
- [Running the API](#running-the-api)  
- [Testing the API](#testing-the-api)  
- [Docker](#docker)  
- [Render Deployment](#render-deployment)  
- [License](#license)

---

## Project Overview

This project delivers a **production-ready machine learning (ML) system** for predicting whether a bank client will subscribe to a term deposit following a marketing campaign. Beyond model development, the focus of this work is on **automation, reproducibility, and deployability**, demonstrating how a data science solution can be engineered for real-world use.

The system spans the **full ML lifecycle** — from exploratory analysis and feature engineering to automated model selection, deployment as a web service, and cloud hosting. The final solution exposes a **FastAPI-based REST API**, containerized with **Docker** and deployed to the cloud using **Render**, making the model accessible for real-time and batch predictions.

### What Makes This Project Production-Ready

Unlike notebook-only machine learning projects, this repository emphasizes engineering discipline and operational readiness:

🔹 Modular ML Pipeline
- Separate notebooks for:
    - Exploratory Data Analysis (EDA)
    - Preprocessing (single source of truth for data splits and feature pipelines)
    - Baseline models
    - Tree-based models
    - Deep learning and anomaly detection
    - Final model evaluation and selection

- Strict train/validation/test isolation to prevent data leakage
- All artifacts (models, metrics, preprocessors) saved and reused consistently

### Automation & Engineering Best Practices
- Artifact versioning: models, metrics, preprocessors saved explicitly
- Environment-aware testing: same client script works locally and in the cloud
- No hard-coded paths: environment variables used for configuration
- Clear separation of concerns:
    - Training ≠ Evaluation ≠ Deployment
- Automated generation of deployment files (Dockerfile, API scaffolding)
- Reproducible experiments via fixed random seeds and saved splits

---

## Project Structure

```text
├── Dockerfile                       # Dockerfile for containerizing the API (can be auto-generated)
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── cleanup_repo.sh                   # Optional script to clean repository files
├── generate_dockerfile.py            # Optional utility script to generate Dockerfile automatically
├── data                              # Folder for raw and processed datasets
├── figures/                          # Saved EDA plots and visualizations
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb       # Data preprocessing pipeline
│   ├── 03_Baseline_Models.ipynb     # Baseline models (LogReg, etc.)
│   ├── 04_Tree_Models.ipynb         # RandomForest & XGBoost
│   ├── 05_Deep_and_Anomaly.ipynb    # MLP & anomaly detection models
│   ├── 06_Model_Evaluation.ipynb    # Final model selection & tuning
│   └── 06_Model_Evaluation.py       # Final model selection & tuning
├── main.py                          # FastAPI application (can be generated automatically)
├── artifacts/
│   ├── preprocessor.pkl              # Preprocessing pipeline
│   └── best_model.json               # Metadata for selected model
├── models/
│   └── best_model.pkl                # Final deployed model
├── test_api.py                       # Local Test API predictions
├── cloud_test_api.py                 # Cloud (Render) Test API predictions
├── api_test.py                       # Automated Test API predictions for both cloud & local
└── sample_requests/
    ├── client.json                   # Single client input
    └── batch_clients.json            # Batch input
```
---

## Business Objective
### Problem Statement

Financial institutions need to identify which clients are likely to subscribe to a bank deposit to **improve campaign efficiency and reduce marketing costs**.

### Goal

Predict whether a client will subscribe to a bank deposit using machine learning.
- The model outputs a probability between 0 and 1 representing subscription likelihood.
- Default threshold = 0.5:
    - `proba >= 0.5` → predicted subscriber
    - `proba < 0.5` → predicted non-subscriber

⚠️ For imbalanced datasets (~11% subscribed), thresholds can be adjusted (e.g., 0.3) to increase recall.

### Why This Project Matters
From a business perspective, predicting customer subscription likelihood enables:
- More efficient marketing campaigns
- Reduced customer fatigue
- Higher conversion rates
- Better allocation of sales resources

From a technical perspective, this project showcases the ability to:
- Design ML systems beyond notebooks
- Make principled metric choices for imbalanced data
- Build and deploy real-time ML services
- Operate at the intersection of data science and software engineering
---

## Dataset
### Source

Kaggle: [Bank Marketing Campaign Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

### Description
- 45,211 client records with demographic, economic, and campaign features.
- Target: y → 1 = subscribed, 0 = not subscribed.
- Features include:
    - `age, job, marital, education`
    - `default, housing, loan`
    - `contact, month, day_of_week, duration`
    - `campaign, pdays, previous, poutcome`

### Why this dataset works
- Public and structured for supervised learning.
- Slight class imbalance (~11% subscribed), suitable for PR-AUC evaluation.
- Can apply various models: RandomForest, XGBoost, Logistic Regression, MLP.

---

## Data Preparation
### Exploratory Data Analysis (EDA)
- Examine class balance.
- Visualize categorical and numerical distributions.
- Inspect correlations between features and target.
### Preprocessing
- Encode categorical features (Label, OneHot, CatBoost).
- Scale numerical features.
- Handle class imbalance with SMOTE or class weights.
- Split dataset into train / validation / test (60/20/20).
- Save splits and preprocessor pipeline for reproducibility.

---

## Modeling Approach
### Models Trained
1. Logistic Regression (LR)
2. Support Vector Machine (SVM)
3. Random Forest (RF)
4. XGBoost
5. Multi-Layer Perceptron (MLP)
6. IsolationForest / One-Class SVM / Autoencoder (anomaly detection)
### Model Selection
- Evaluated all models using PR-AUC as primary metric.
    - Focuses on positive class performance.
    - Ideal when the dataset is imbalanced, i.e., the number of positive cases (subscribers/fraud) is much smaller than negative cases.
    - Measures how well the model balances precision and recall across thresholds.
    - Why it’s ideal here: In the bank marketing dataset or fraud detection, only a small fraction subscribe, so PR AUC gives a more realistic view of performance than accuracy.

- Top 2 models tuned with Optuna.
- Final model automatically selected based on PR-AUC on test set.

### Deployment Choice
- XGBoost selected and deployed as production model.
---
## Model Evaluation

Metrics used (focus on subscription detection):
- PR-AUC (Precision–Recall AUC) – Primary metric
- ROC-AUC – Secondary metric
- F1 Score, Precision, Recall – for interpretability
- Thresholds adjustable depending on campaign priorities


## Metrics Comparison

| Model              | roc_auc | pr_auc  | f1      | precision | recall  | threshold |
|------------------- |--------:|--------:|--------:|----------:|--------:|----------:|
| logreg             | 0.9235  | 0.5606  | 0.5931  | 0.4537    | 0.8562  | 0.5000    |
| random_forest      | 0.9310  | 0.6066 ✅ | 0.5746  | 0.5807    | 0.5687  | 0.5000    |
| xgboost            | 0.9334 ✅ | 0.6033  | 0.5844 ✅ | 0.5954 ✅ | 0.5738  | 0.5000    |
| mlp                | 0.9078  | 0.5308  | 0.3995  | 0.2503    | 0.9883 ✅ | 0.0000    |
| isolation_forest   | 0.5929  | 0.1883  | 0.2471  | 0.1549    | 0.6114  | -0.0447   |
| one_class_svm      | 0.5904  | 0.1746  | 0.2513  | 0.1575    | 0.6218  | -26.2506  |
| autoencoder        | 0.6289  | 0.2093  | 0.2686  | 0.1683    | 0.6645  | 0.0050    |

**Legend:**  
- ✅ Best value per metric  
- `roc_auc` = Area under the ROC curve  
- `pr_auc` = Area under the Precision-Recall curve  
- `f1` = F1-score  
- `precision`, `recall` = standard classification metrics  
- `threshold` = probability threshold used for prediction

### Final Best Model After Optuna Hyperparameter Tuning

#### Model: XGBoost

**Metrics on Test Data:**

| Metric      | Value  |
|------------ |-------:|
| roc_auc     | 0.9379 ✅ |
| pr_auc      | 0.6347 ✅ |
| f1          | 0.6317 ✅ |
| precision   | 0.5742  |
| recall      | 0.7021  |
| threshold   | 0.5000  |

**Notes:**  
- ✅ Indicates the best performance compared to previous models.  
- Threshold = probability cutoff used for binary classification.  
- Metrics were obtained on the held-out test dataset after hyperparameter tuning with Optuna.
--- 

## Deployment Workflow
### Model Serialization
- Preprocessor and final model saved using joblib.
- Model metadata stored in artifacts/best_model.json.

### API Development
- FastAPI service with endpoints:
    - POST /predict → single client prediction
    - POST /predict_batch → batch predictions

### Containerization
- Fully containerized API using Docker
- Identical behavior across local, container, and cloud environments
- Enables seamless deployment and portability
- Image runs FastAPI for real-time predictions

### Cloud Deployment (Render)
- Deployed as a live cloud service
- Publicly accessible REST endpoint
- Supports automated testing against production
- Environment-aware client testing (local vs cloud)


---

## Requirements
- Python 3.10+
- Pip / Conda

Install dependencies
```text
pip install -r requirements.txt
```
---

## Setup & Installation
Clone the repository:
```text
git clone https://github.com/o-izima/bank-deposit-subscription-prediction.git
cd bank-deposit-subscription-prediction
```

Install dependencies:
```text
pip install -r requirements.txt
```

Ensure `models/` contains `best_model.pkl` and `artifacts/` contains `preprocessor.pkl` and `best_model.json`.

---

## Running the API Locally

Start the FastAPI app: `python main.py`

The API will run at:
`http://localhost:8000`

### Endpoints:
- POST /predict → predict subscription for single client
- POST /predict_batch → batch predictions
- GET / → health check

## Testing the API

Use `api_test.py` with sample JSON files.
- The script `api_test.py` auto-detects the environment so it works for both local and cloud deployments.
- It uses an environment variable called `API_ENV`.
- `API_ENV=local` → uses http://localhost:8000
- `API_ENV=cloud` → uses your Render URL

Example files:
- `sample_requests/client.json` – single client
- `sample_requests/batch_clients.json` – batch of clients

The script prints predicted subscription probabilities.

### Usage
#### Local Testing (ensure docker is runnning)
```text
export API_ENV=local
python api_test.py
```

#### Cloud Testing
```text
export API_ENV=cloud
python api_test.py
```

#### Sample Output
```text
Using API URL: https://bank-deposit-subscription-api.onrender.com (Environment: cloud)

--- Single Client Prediction ---
{'model': 'xgboost', 'probability': 0.0162, 'prediction': 0}
Predicted Class: 0
Subscription Probability: 0.0162

--- Batch Predictions ---
Model Used: xgboost
1. Prediction: 0, Probability: 0.0214
2. Prediction: 1, Probability: 0.6821

```
---
## Docker
Build the Docker image:
```text
docker build -t bank-marketing-api .
```
Run the Docker container locally:
```text
docker run -p 8000:8000 bank-marketing-api
```
---

## Render Deployment
Render Cloud URL: [Bank Marketing Subscription Prediction API](https://bank-deposit-subscription-api.onrender.com)

---

## License

This project is licensed under the [MIT License](./LICENSE). See the LICENSE file for details.

[⬆ Go to Top](#top)