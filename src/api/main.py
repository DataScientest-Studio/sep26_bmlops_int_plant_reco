"""API for the plant recognition model."""
from fastapi import FastAPI, File, UploadFile

from src.data.etl import run_etl
from src.models.predict import predict
from src.models.training import train


from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Histogram
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Plant Reco API")


# 1. Automatically instrument standard HTTP metrics 
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# 2. Define the custom prediction confidence histogram metric
PREDICTION_CONFIDENCE = Histogram(
    name="prediction_confidence",
    documentation="Histogram of model prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)







@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/etl")
def etl_endpoint(per_class: int = 20):
    """Add the next batch of new images (per class) to the database."""
    return run_etl(per_class)


@app.post("/train_evaluate")
@app.post("/training")
def train_evaluate_endpoint():
    """Train a new model, log it in MLflow and compare it with the production model."""
    return {"message": "Model trained, evaluated and registered in MLflow", "result": train()}


@app.post("/predict")
def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = file.file.read()

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".jfif")):
        raise HTTPException(
            status_code=500, detail="Internal server error: Invalid file format."
        )

    image_bytes = file.file.read()
    result = predict(image_bytes)

    # Extract confidence score from prediction result and record it in the histogram
    confidence = 0.5
    if isinstance(result, dict):
        confidence = result.get("confidence", result.get("score", 0.5))

    PREDICTION_CONFIDENCE.observe(confidence)

    return result
    
