from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title ="PRCF Reservation ML Model API")
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/best_model_pipeline.pkl"))


try:
    model= joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f" Could not load model:{e}")
    model = None


#input schema
class InputData(BaseModel):
    features: list[float]


#routes
@app.get("/")
def root():
    return {"message": "PRCF ML model is ready to serve predictions"}


@app.get("/health")
def health():
    return {"status": "ok" if model else "model not loaded", "model_path": MODEL_PATH}


@app.post("/predict")
def predict(data:InputData):
    if  model is None:
        return {"error":"Model not loaded"}
    if not data.features:
        return {"error": "Feature list is empty"}
    

    #return a prediction given feature list
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": prediction.tolist()}


