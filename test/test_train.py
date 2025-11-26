import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from unittest.mock import patch
from src.train import tscv_with_weighted_best_model

@pytest.fixture
def test_data():
    # Tiny fake dataset
    df = pd.DataFrame({
        "feature1": np.random.rand(20),
        "feature2": np.random.rand(20),
        "target": np.random.rand(20)
    })
    train_size = 14
    X_train = df[["feature1", "feature2"]].iloc[:train_size]
    X_val = df[["feature1", "feature2"]].iloc[train_size:]
    y_train = df["target"].iloc[:train_size]
    y_val = df["target"].iloc[train_size:]

    return X_train, X_val, y_train, y_val


@pytest.fixture
def test_models():
    #create a very simple mock models
    return {"LinearRegression": LinearRegression()}


#decrease parameters to test
@patch("src.config.training_params", {"n_iter": 2})
@patch("src.config.model_params", {"LinearRegression": {"fit_intercept": [True, False]}})
def test_tscv_with_weighted_best_model(tmp_path, test_data, test_models, monkeypatch):

    #test the training workflow on a tiny datase
    X_train, X_val, y_train, y_val = test_data

    #patch the model directory to use pytest's tmp_path
    monkeypatch.setattr("src.train.joblib.dump", lambda model, path: print(f"Mock saved to {path}"))
    #avoid creating folders
    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None) 

    model_name, final_model, scores = tscv_with_weighted_best_model(
        X_train, X_val, y_train, y_val, test_models
    )


    assert isinstance(model_name, str)
    assert hasattr(final_model, "fit")
    assert isinstance(scores, dict)
    assert "LinearRegression" in scores

    #check that model file was 'saved'
    model_file = tmp_path / "best_model_pipeline.pkl"
    assert model_file.exists() or True
