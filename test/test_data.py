import pytest
import pandas as pd
from src.data import load_data
from src.config import client_params

def test_load_data_real_small_sample():
    """Test the real Socrata connection, but only download a few records."""
    
    # Create a smaller version of the config for testing
    small_params = client_params.copy()
    small_params["limit"] = 5

    # Call the same function but with smaller params
    df = load_data(custom_params=small_params)

    # Verify the results
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert len(df) <= 5

