
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def get_models_to_use():

    models={'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42),        
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'RandomForestRegressor': RandomForestRegressor(random_state=42, n_jobs=-1),
            'XGBRegressor': XGBRegressor(tree_method='gpu_hist', gpu_id=0, predictor='gpu_predictor', n_jobs=-1)
            }

    return models

