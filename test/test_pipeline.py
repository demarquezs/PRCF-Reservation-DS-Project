from unittest.mock import patch, MagicMock
from src.pipeline import (load_data_task, obtain_models_task, training_model_task, mlops_pipeline_flow)


#test load_data_task
def test_load_data_task(monkeypatch):
    #ensure the data loading task correctly chains the functions
    with patch("src.pipeline.load_data", return_value="fake_raw"), \
         patch("src.pipeline.filter_and_process_data", return_value="fake_filtered"), \
         patch("src.pipeline.transform_and_split_data", return_value=(1, 2, 3, 4)):

        result = load_data_task.fn()  
        
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert result == (1, 2, 3, 4)


#test obtain_models_task
def test_obtain_models_task():
    #ensure models are returned as a dictionary with expected keys
    
    result = obtain_models_task.fn()
    
    assert isinstance(result, dict)
    assert "RandomForestRegressor" in result
    assert "LinearRegression" in result


#test training_model_task
@patch("src.pipeline.mlflow.sklearn.log_model")
@patch("src.pipeline.mlflow.start_run")
def test_training_model_task(mock_log_model, mock_start_run):

    #ensure the training task calls MLflow and returns expected outputs
    X_train, X_val, y_train, y_val = [1], [2], [3], [4]
    fake_model = MagicMock()
    fake_models = {"LinearRegression": fake_model}

    with patch("src.pipeline.tscv_with_weighted_best_model",
               return_value=("LinearRegression", fake_model, {"LinearRegression": 0.95})):

        best_name, best_model, scores = training_model_task.fn(X_train, X_val, y_train, y_val, fake_models)

        assert best_name == "LinearRegression"
        assert scores["LinearRegression"] == 0.95
        mock_log_model.assert_called_once()



#test mlops_pipeline_flow (integration)
@patch("src.pipeline.load_data_task.fn", return_value=("X_train", "X_val", "y_train", "y_val"))
@patch("src.pipeline.obtain_models_task.fn", return_value={"RandomForestRegressor": MagicMock()})
@patch("src.pipeline.training_model_task.fn",return_value=("RandomForestRegressor", MagicMock(), {"RandomForestRegressor": 0.9}))

def test_mlops_pipeline_flow(mock_train, mock_models, mock_data):
    #ensure the full pipeline flow runs end-to-end without errors."""
    result = mlops_pipeline_flow()

    #it should return a tuple: (best_name, best_model, weighted_scores)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "RandomForestRegressor"
    assert "RandomForestRegressor" in result[2]

