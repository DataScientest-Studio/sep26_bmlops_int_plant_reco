"""API for the plant recognition model."""
from fastapi import FastAPI, File, UploadFile

from src.models.predict import predict
from src.models.training import train

app = FastAPI(title="Plant Reco API")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/train_evaluate")
@app.post("/training")
def train_evaluate_endpoint():
    """Train a new model, log it in MLflow and compare it with the production model."""
    return {"message": "Model trained, evaluated and registered in MLflow", "result": train()}


@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    return predict(image_bytes)
