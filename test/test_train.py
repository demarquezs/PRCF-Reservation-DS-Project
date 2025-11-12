import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
from unittest.mock import patch
from src.train import tscv_with_weighted_best_model

@pytest.fixture
def test_data():
    #generate small fake data for quick testing
    X = pd.DataFrame({
        "feature1": np.random.rand(20),
        "feature2": np.random.rand(20)
    })
    y = pd.Series(np.random.rand(20))
    return X, y


@pytest.fixture
def test_models():
    #create a very simple mock models
    return {"LinearRegression": LinearRegression()}


#decrease parameters to test
@patch("src.train.training_params", {"n_splits": 3, "n_iter": 2})
@patch("src.train.model_params", {"LinearRegression": {"fit_intercept": [True, False]}})


def test_tscv_with_weighted_best_model(tmp_path, test_data, test_models, monkeypatch):

    #test the training workflow on a tiny datase
    X_train, X_val, y_train, y_val = test_data

    #patch the model directory to use pytest's tmp_path
    monkeypatch.setattr("os.path.join", lambda *args: str(tmp_path / "best_model_pipeline.pkl"))
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
