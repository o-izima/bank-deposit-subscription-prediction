<a name="top"></a>
# Bank Deposit Subscription Prediction – Production-Ready ML Pipeline with FastAPI, Docker, and Render Deployment

This machine learning project predicts whether a client will subscribe to a bank deposit campaign using XGBoost, the best-performing model among all models trained. The project provides Jupyter notebooks for EDA, preprocessing, and model training, along with a FastAPI-based REST API deployed both locally via Docker and in the cloud on Render, enabling real-time subscription predictions.

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
- [License](#license)

---

## Project Overview

Banks frequently run marketing campaigns to attract new deposits from clients. This project builds a complete machine learning system capable of:

- Predicting whether a client will subscribe to a term deposit campaign.  
- Producing probability-based subscription scores.  
- Deploying the best-performing model as a REST API using FastAPI and Docker.

The project covers the **full ML lifecycle** — from EDA and preprocessing to model selection, hyperparameter tuning, and deployment.

---

## Project Structure

```text
.
├── Dockerfile                       # Dockerfile for containerizing the API
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
├── notebooks/
│   ├── 01_EDA.ipynb                 # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb       # Data preprocessing pipeline
│   ├── 03_Baseline_Models.ipynb     # Baseline models (LogReg, etc.)
│   ├── 04_Tree_Models.ipynb         # RandomForest & XGBoost
│   ├── 05_Deep_and_Anomaly.ipynb    # MLP & anomaly detection models
│   ├── 06_Model_Evaluation.ipynb    # Final model selection
│   └── 07_Explainability.ipynb      # SHAP / feature importance
├── app/
│   ├── main.py                       # FastAPI application
│   ├── predict.py                    # Model inference logic
│   └── schemas.py                    # Pydantic request/response models
├── artifacts/
│   ├── preprocessor.pkl              # Preprocessing pipeline
│   └── best_model.json               # Metadata for selected model
├── models/
│   └── best_model.pkl                # Final deployed model
├── test_api.py                       # Test API predictions
└── sample_requests/
    ├── client.json                   # Single client input
    └── batch_clients.json            # Batch input

