"""Predict the class of one leaf image with the production model from MLflow."""
import sys

import mlflow.sklearn
from mlflow import MlflowClient

from src.models.promote import ALIAS, MODEL_NAME
from src.models.training import preprocess


def predict(image_bytes):
    """Return the predicted class, the confidence and the model version that answered."""
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    x = preprocess(image_bytes).reshape(1, -1)
    label = model.predict(x)[0]
    confidence = float(model.predict_proba(x).max())
    return {"label": str(label), "confidence": round(confidence, 4), "model_version": int(version)}


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        print(predict(f.read()))
