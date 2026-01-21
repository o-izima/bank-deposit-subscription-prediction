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
- [Model Metrics Comparison](#model-metrics-comparison)  
- [Deployment Workflow](#deployment-workflow)  
- [Requirements](#requirements)  
- [Technologies](#technologies)  
- [Setup & Installation](#setup--installation)  
- [Running the API](#running-the-api)  
- [Testing the API](#testing-the-api)  
- [Docker](#docker)  
- [Render Deployment](#render-deployment)  
- [License](#license)

---

## Project Overview

Banks frequently run marketing campaigns to attract new deposits from clients. This project builds a **production-ready machine learning system** capable of:

- Predicting whether a client will subscribe to a term deposit campaign.  
- Producing probability-based subscription scores.  
- Deploying the best-performing model via FastAPI **locally and in the cloud using Render**.

The project covers the **full ML workflow** — from EDA and preprocessing to model selection, hyperparameter tuning, and deployment.

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
│   ├── 06_Model_Evaluation.ipynb    # Final model selection
├── main.py                          # FastAPI application (can be generated automatically)
├── artifacts/
│   ├── preprocessor.pkl              # Preprocessing pipeline
│   └── best_model.json               # Metadata for selected model
├── models/
│   └── best_model.pkl                # Final deployed model
├── test_api.py                       # Local Test API predictions
├── cloud_test_api.py                 # Cloud (Render) Test API predictions
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

### Impact
- Focus marketing efforts on clients likely to subscribe.
- Reduce cost of outreach.
- Improve campaign ROI.
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


## 📊 Model Metrics Comparison

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
