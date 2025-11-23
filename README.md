# 🏞️ PRCF Reservation

## 📊 Project Overview

**PRCF Reservation** is a comprehensive **data science project** analyzing Parks, Recreation and Community Facilities (PRCF) data from Mesa, Arizona. This project leverages open government data to understand visitation patterns and forecast future attendance levels at various parks and tourist centers throughout Mesa.

---

## ✨ Key Features

* **Data Extraction & Exploration:** Comprehensive sourcing from government open data portals.
* **SQL-Based EDA:** In-depth exploratory data analysis using SQL queries.
* **Time Series Forecasting:** Advanced machine learning models to predict annual attendance levels.
* **Comparative Model Analysis:** Evaluation of different forecasting approaches.

---

## 📁 Project Structure

The project consists of the following modular components:

```bash
PRCF_Reservation/
│
├── src/                      # Main pipeline logic
│   ├── data.py               # Data ingestion and cleaning
│   ├── model.py              # Model registry and hyperparameters
│   ├── train.py              # TimeSeriesSplit training and evaluation
│   ├── pipeline.py           # Prefect orchestration pipeline
│   ├── config.py             # Centralized configuration
│   └── utils.py              # Utility functions
│
├── api/                      # FastAPI app for model inference
│   ├── main.py               # API routes and prediction endpoint
│   ├── requirements.txt      # API dependencies
│   └── Dockerfile            # Container for serving model
│
├── models/                   # Trained model artifacts (excluded from Git)
├── notebooks/                # Experimental analysis and EDA notebooks
├── requirements_data.txt     # Dependencies for data processing
├── requirements_mlops.txt    # Dependencies for MLOps stack
├── prefect.yaml              # Prefect deployment configuration
└── .github/workflows/mlops_pipeline.yml   # CI/CD automation
```

---

## ⚙️ MLOps Workflow

After the Data Science phase, the project evolved into a **production-ready MLOps pipeline**. The workflow integrates **Prefect**, **GitHub Actions**, **Docker**, and **MLflow**.

### **1️⃣ Prefect Pipeline**

* Orchestrates tasks such as **data loading**, **model training**, and **artifact logging**.
* Runs locally or in **Prefect Cloud** using a defined deployment.
* Automatically logs parameters and metrics.

```bash
prefect cloud login -k <API_KEY>
prefect deploy src/pipeline.py:mlops_pipeline_flow -n "production" -q "default"
```

### **2️⃣ MLflow Integration**

* Logs training performance, parameters, and models.
* Tracks each model version during CI/CD runs.

```python
import mlflow
mlflow.log_metric("r2_score", model_score)
mlflow.sklearn.log_model(model, "model")
```

### **3️⃣ Dockerization**

Two separate Dockerfiles manage different stages:

* **Training Image:** runs the Prefect pipeline locally or in GitHub Actions.
* **API Image:** exposes the trained model with FastAPI using Uvicorn.

```Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/main.py .
COPY models /app/models
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### **4️⃣ GitHub Actions CI/CD**

Automates the complete MLOps lifecycle:

* Triggers the Prefect pipeline on every push.
* Uploads the trained model as an artifact.
* Builds and pushes the FastAPI Docker image to Docker Hub.
* Triggers Prefect deployment.

```yaml
name: MLOps Pipeline
on:
  push:
    branches: [ main ]
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run pipeline
        run: python -m src.pipeline
  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: docker/build-push-action@v5
```

---

## ☁️ Prefect Cloud & Monitoring

* All training and orchestration logs are visible from **Prefect Cloud UI**.
* You can trigger, reschedule, or monitor flows in real time.
* Integrates easily with alerts, task retries, and versioning.

---

## 🧠 Optimization Recommendations

| **Area**                | **Recommendation**                                                            |
| ----------------------- | ----------------------------------------------------------------------------- |
| **Storage**             | Use AWS S3 or MLflow Artifact Store for large `.pkl` files instead of GitHub. |
| **Testing**             | Add `pytest` and linting (flake8/black) to GitHub Actions.                    |
| **Prefect Reliability** | Add caching, retries, and notifications to tasks.                             |
| **Docker Optimization** | Apply multi-stage builds for smaller images.                                  |
| **Deployment**          | Host the API container on AWS ECS, Azure Container Apps, or GCP Cloud Run.    |

---

## 🎯 Outcome

✅ Began as a **data science forecasting project**.
✅ Evolved into a **fully automated MLOps pipeline**.
✅ Integrated **Prefect**, **MLflow**, **Docker**, and **GitHub Actions**.
✅ Ready for **continuous training, deployment, and monitoring**.

---

**👨‍💻 Author:** David Sánchez
**🎯 Goal:** Demonstrate the transition from **Data Science experimentation** to **MLOps automation and deployment**.
