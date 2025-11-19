import pytest
import numpy as np
from src.model import get_models_to_use


def test_get_models_to_use_returns_dict():
    #Check that get_models_to_use returns a dictionary of models
    models = get_models_to_use()

    assert isinstance(models, dict)
    assert len(models) > 0
