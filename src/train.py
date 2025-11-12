
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from src.config import training_params, model_params
from sklearn.metrics import r2_score
import numpy as np
import time
import joblib
import os

#create a function that standardizes the scale, divide data ofr train and test, trains models, splits the data for time series giving weights to the splits (giving more weight to the most recent splits), evaluates the models using these splits and selects (and save) the best model:
def tscv_with_weighted_best_model(X_train_scaled, X_val, y_train, y_val, models):

    n_splits = training_params['n_splits']
    n_iter = training_params['n_iter']
    param_distributions = model_params

    #initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    weights = np.linspace(0, 1, num=n_splits) 
    #initialize model performance tracking  
    
    best_models = {}
    weighted_scores = {}

    start_time = time.time()
    
    print("Starting models training and evaluation with TimeSeriesSplit")
    
    #train and evaluate each model with RandomizedSearchCV and TimeSeriesSplit
    
    for name, model in models.items():
        print(f"\nTraining and Evaluating {name}:")

        search = RandomizedSearchCV(model, param_distributions=param_distributions[name],
                                    n_iter=n_iter, scoring='r2', cv=tscv, n_jobs=-1)

        search.fit(X_train_scaled, y_train)
        best_models[name] = search.best_estimator_

        print(f"Best params for {name}: {search.best_params_}")
         

        split_scores = []
        for i, (train_idx, test_idx) in enumerate(tscv.split(X_val)):
            X_val_train, X_val_test = X_val.iloc[train_idx], X_val.iloc[test_idx]
            y_val_train, y_val_test = y_val.iloc[train_idx], y_val.iloc[test_idx]
            best_model = best_models[name]  
            best_model.fit(X_val_train, y_val_train)
            preds = best_model.predict(X_val_test)
            score = r2_score(y_val_test, preds)
            split_scores.append(score * weights[i])
    
        weighted_scores[name] = np.sum(split_scores) / np.sum(weights)
        print(f"Weighted R² Score for {name}: {weighted_scores[name]:.4f}")
    
    
    final_model_name = max(weighted_scores, key=weighted_scores.get)
    final_model = best_models[final_model_name]
    
    end_time = time.time()
    print(f"\nFinal selected model: {final_model_name} with weighted R² Score: {weighted_scores[final_model_name]:.4f}")
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    os.makedirs(model_dir, exist_ok=True)

    final_model_path = os.path.join(model_dir, "best_model_pipeline.pkl")
    joblib.dump(final_model, final_model_path)
    print(f"✅ Model saved in: {final_model_path}")

    # Debug: confirm file exists
    print("Files in model directory:", os.listdir(model_dir))



    return final_model_name, final_model, weighted_scores
