
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.utils import detect_gpu


def get_models_to_use():
    
    use_gpu = detect_gpu()
    device = 'cuda' if use_gpu else 'cpu'

    print(f"Initializing models with device: {device}")

    models={'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42),        
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'RandomForestRegressor': RandomForestRegressor(random_state=42, n_jobs=-1),
            'XGBRegressor': XGBRegressor(tree_method='hist', device=device, 
                                         objective='reg:squarederror', n_jobs=-1, 
                                         random_state=42)
            }

    return models

