"""API for the plant recognition model."""
from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.data.etl import run_etl
from src.features.build_features import image_features
from src.models.predict import predict
from src.models.training import train
from src.monitoring.drift import (
    check_image_drift,
    check_prediction_drift,
)

app = FastAPI(title="Plant Reco API")

# HTTP metrics (requests, status codes, latency) on GET /metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Confidence of every prediction, for the alert in Grafana
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence of the model's predictions",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Brightness and sharpness of every image, for the drift alerts
IMAGE_BRIGHTNESS = Histogram(
    "image_brightness",
    "Mean brightness of the uploaded images (0 to 255)",
    buckets=[25, 50, 75, 100, 125, 150, 175, 200, 225, 255],
)
IMAGE_SHARPNESS = Histogram(
    "image_sharpness",
    "Mean sharpness of the uploaded images",
    buckets=[1, 2, 3, 5, 8, 10, 12, 15, 20, 30],
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
    """Predict the class of a leaf image, record its metrics."""
    if not file.filename.lower().endswith(IMAGE_TYPES):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .jpg or .png image",
        )
    image_bytes = file.file.read()
    result = predict(image_bytes)
    features = image_features(image_bytes)
    PREDICTION_CONFIDENCE.observe(result["confidence"])
    IMAGE_BRIGHTNESS.observe(features["brightness"])
    IMAGE_SHARPNESS.observe(features["sharpness"])
    return result


@app.post("/drift/images")
def drift_images_endpoint():
    """Compare the newest images with the first load (Evidently)."""
    return check_image_drift()


@app.post("/drift/predictions")
def drift_predictions_endpoint(n: int = 30):
    """Compare the n latest predictions with the first load."""
    return check_prediction_drift(n)
