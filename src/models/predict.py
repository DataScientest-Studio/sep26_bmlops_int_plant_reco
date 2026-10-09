"""Predict the class of one leaf image with the saved model."""
import sys

import joblib

from src.models.training import MODEL_PATH, preprocess


def predict(image_bytes):
    """Return the predicted class and the model's confidence for one image."""
    model = joblib.load(MODEL_PATH)
    x = preprocess(image_bytes).reshape(1, -1)
    label = model.predict(x)[0]
    confidence = float(model.predict_proba(x).max())
    return {"label": str(label), "confidence": round(confidence, 4)}


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        print(predict(f.read()))
