"""API for the plant recognition model."""
from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.data.etl import run_etl
from src.models.predict import predict
from src.models.training import train

app = FastAPI(title="Plant Reco API")

# HTTP metrics (requests, status codes, latency) on GET /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Confidence of every prediction, for the alert in Grafana
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence of the model's predictions",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

IMAGE_TYPES = (".jpg", ".jpeg", ".png")


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/etl")
def etl_endpoint(per_class: int = 20):
    """Add the next batch of new images to the database."""
    return run_etl(per_class)


@app.post("/train_evaluate")
@app.post("/training")
def train_evaluate_endpoint():
    """Train, log in MLflow, compare with the production model."""
    return {
        "message": "Model trained, evaluated and registered in MLflow",
        "result": train(),
    }


@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    """Predict the class of a leaf image, record its confidence."""
    if not file.filename.lower().endswith(IMAGE_TYPES):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .jpg or .png image",
        )
    result = predict(file.file.read())
    PREDICTION_CONFIDENCE.observe(result["confidence"])
    return result
