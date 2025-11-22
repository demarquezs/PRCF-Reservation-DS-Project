import prefect
from prefect import flow, task
from logging import config
from src.data import load_data, filter_and_process_data, transform_and_split_data
from src.model import get_models_to_use
from src.train import tscv_with_weighted_best_model
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import os
import numpy as np


prefect.settings.PREFECT_API_SERVICES_ENABLED = True
prefect.settings.PREFECT_API_URL = "http://127.0.0.1:8219/api"


@task
def load_data_task():

    #load the data
    df = load_data()
    df_filtered = filter_and_process_data(df)
    X_train_scaled, X_val, y_train, y_val = transform_and_split_data(df_filtered)

    return X_train_scaled, X_val, y_train, y_val


@task
def obtain_models_task():

    #call the models
    models= get_models_to_use()

    return models


@task
def training_model_task(X_train_scaled, X_val, y_train, y_val, models):
    #train the model
    with mlflow.start_run(run_name="Training_All_Models"):
        
        #onecall function handles all models
        best_name, best_model, weighted_scores = tscv_with_weighted_best_model(X_train_scaled, X_val, y_train, y_val, models)

        #log weighted R² for every model
        for mname, score in weighted_scores.items():
            mlflow.log_metric(f"{mname}_weighted_r2", float(score))

        print("Calling MLflow log_model for:", best_name)

        #create signature from training data
        if hasattr(X_train_scaled, 'shape'):
            sample_input = X_train_scaled.iloc[:5]
        else:
            sample_input = X_train_scaled[:5]
        
        sample_output = best_model.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        #log best model to MLflow with signature
        mlflow.sklearn.log_model( best_model, artifact_path="model", signature=signature, registered_model_name=best_name)


    return best_name, best_model, weighted_scores


@flow
def mlops_pipeline_flow():

    #delete old cache to force fresh download with new limit
    cache_path = "/home/ingde/Projects/data/raw/PRCF_reservation.csv"
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Deleted old cache: {cache_path}")

    mlflow.set_experiment("MLOps_Project_TimeSeries")

    X_train_scaled, X_val, y_train, y_val = load_data_task()
    models = obtain_models_task()
    best_name, best_model, weighted_scores= training_model_task(X_train_scaled, X_val, y_train, y_val, models)

    return best_name, best_model, weighted_scores

if __name__ == "__main__":
    mlops_pipeline_flow()