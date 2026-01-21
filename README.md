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
- [Metrics Table](#metrics-table)  
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
