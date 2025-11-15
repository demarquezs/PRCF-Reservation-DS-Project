
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from src.config import training_params, model_params
from sklearn.metrics import r2_score
import numpy as np
import time
import joblib
import os

#create a function that standardizes the scale, divide data ofr train and test, trains models, splits the data for time series giving weights to the splits (giving more weight to the most recent splits), evaluates the models using these splits and selects (and save) the best model:
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score
import numpy as np
import joblib
import time
import os


def tscv_with_weighted_best_model(X_train_scaled, X_val, y_train, y_val, models):

    n_splits = training_params["n_splits"]
    n_iter = training_params["n_iter"]
    param_distributions = model_params

    tscv = TimeSeriesSplit(n_splits=n_splits)
    weights = np.linspace(0, 1, num=n_splits)

    best_estimators = {}
    weighted_scores = {}

    start_time = time.time()
    print("Starting models training and evaluation with TimeSeriesSplit")

    #train each model
    for name, model in models.items():
        print(f"\nTraining and Evaluating {name}...")

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[name],
            n_iter=n_iter,
            scoring="r2",
            cv=tscv,
            n_jobs=-1
        )

        search.fit(X_train_scaled, y_train)
        best_estimators[name] = search.best_estimator_

        print(f"Best params for {name}: {search.best_params_}")


        #evaluate weighted R² on validation set
        split_scores = []

        for i, (train_idx, test_idx) in enumerate(tscv.split(X_val)):
            X_val_train, X_val_test = X_val.iloc[train_idx], X_val.iloc[test_idx]
            y_val_train, y_val_test = y_val.iloc[train_idx], y_val.iloc[test_idx]

            model = best_estimators[name]
            model.fit(X_val_train, y_val_train)
            preds = model.predict(X_val_test)

            score = r2_score(y_val_test, preds)
            weighted = score * weights[i]
            split_scores.append(weighted)

        weighted_scores[name] = np.sum(split_scores) / np.sum(weights)
        print(f"Weighted R² for {name}: {weighted_scores[name]:.4f}")


    #select best model
    final_model_name = max(weighted_scores, key=weighted_scores.get)
    final_model = best_estimators[final_model_name]

    print(f"\nFinal selected model: {final_model_name} "
          f"with weighted R²: {weighted_scores[final_model_name]:.4f}")


    #save model
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    os.makedirs(model_dir, exist_ok=True)

    final_model_path = os.path.join(model_dir, "best_model_pipeline.pkl")
    joblib.dump(final_model, final_model_path)

    print(f"✅ Model saved at: {final_model_path}")
    print("Files in the model directory:", os.listdir(model_dir))

    print(f"Total training time: {time.time() - start_time:.2f} seconds")

    return final_model_name, final_model, weighted_scores

