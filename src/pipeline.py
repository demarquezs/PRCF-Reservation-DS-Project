from prefect import flow, task
from logging import config
from src.data import load_data, filter_and_process_data, transform_and_split_data
from src.model import get_models_to_use
from src.train import tscv_with_weighted_best_model
import mlflow


@task
def load_data_task():

    df = load_data()
    df_filtered = filter_and_process_data(df)
    X_train_scaled, X_val, y_train, y_val = transform_and_split_data(df_filtered)

    return X_train_scaled, X_val, y_train, y_val


@task
def obtain_models_task():

    models= get_models_to_use()

    return models


@task
def training_model_task(X_train_scaled, X_val, y_train, y_val, models):

    with mlflow.start_run(run_name="Training_All_Models") as parent_run:
        # Call once: tscv handles all models internally
        best_name, best_model, weighted_scores = tscv_with_weighted_best_model(X_train_scaled, X_val, y_train, y_val, models)


        # log the weighted scores for every model
        for mname, score in weighted_scores.items():
            mlflow.log_metric(f"{mname}_weighted_r2", float(score))

        # log model artifact
        try:
            mlflow.sklearn.log_model(best_model, artifact_path=f"models/{best_name}")
        except Exception as e:
            print(f"Model logging failed: {e}")

    return best_name, best_model, weighted_scores
    

@flow
def mlops_pipeline_flow():

    mlflow.set_experiment("MLOps_Project_TimeSeries")

    X_train_scaled, X_val, y_train, y_val = load_data_task()
    models = obtain_models_task()
    best_name, best_model, weighted_scores= training_model_task(X_train_scaled, X_val, y_train, y_val, models)

    return best_name, best_model, weighted_scores

if __name__ == "__main__":
    mlops_pipeline_flow()