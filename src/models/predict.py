"""Predict the class of one leaf image with the production model from MLflow."""
import sys

import mlflow.sklearn
from mlflow import MlflowClient

from src.data.predictions import save_prediction
from src.features.build_features import image_features
from src.models.promote import ALIAS, MODEL_NAME
from src.models.training import preprocess



def predict(image_bytes):
    """Return the prediction and save it with its image features."""
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    x = preprocess(image_bytes).reshape(1, -1)
    label = model.predict(x)[0]
    confidence = float(model.predict_proba(x).max())

    result = {
        "label": str(label),
        "confidence": round(confidence, 4),
        "model_version": int(version),
    }

    features = image_features(image_bytes)
    save_prediction(result, features)

    return result


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        print(predict(f.read()))
