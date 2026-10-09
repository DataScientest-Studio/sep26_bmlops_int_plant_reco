"""API for the plant recognition model."""
from fastapi import FastAPI, File, UploadFile

from src.models.predict import predict
from src.models.training import train

app = FastAPI(title="Plant Reco API")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/training")
def training_endpoint():
    metrics = train()
    return {"message": "Model trained and saved", "metrics": metrics}


@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    return predict(image_bytes)
