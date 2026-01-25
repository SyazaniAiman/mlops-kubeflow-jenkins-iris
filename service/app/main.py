import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/iris_model.joblib")

app = FastAPI(
    title="Iris Inference API",
    version=os.getenv("APP_VERSION", "0.0.0")
)

CLASS_NAMES = ["setosa", "versicolor", "virginica"]

class IrisFeatures(BaseModel):
    features: List[float] = Field(..., min_length=4, max_length=4)

class PredictionOut(BaseModel):
    predicted_class: int
    class_name: str

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python service/train_local.py"
        )
    return joblib.load(MODEL_PATH)

model = None

def get_model():
    global model
    if model is None:
        model = load_model()
    return model

@app.on_event("startup")
def startup_event():
    # Keep startup load for normal running
    get_model()


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: IrisFeatures):
    m = get_model()
    x = [payload.features]
    y = int(m.predict(x)[0])
    return {"predicted_class": y, "class_name": CLASS_NAMES[y]}

