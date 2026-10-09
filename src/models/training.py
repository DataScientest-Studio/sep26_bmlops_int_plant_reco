"""Train a baseline model on the images stored in the database."""
import io
import sqlite3
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DB_PATH = Path("data/plants.db")
MODEL_PATH = Path("models/model.joblib")
IMG_SIZE = 32
SEED = 42


def preprocess(image_bytes):
    """Image bytes -> flat vector of 32*32*3 numbers between 0 and 1."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32).flatten() / 255.0


def load_data():
    """Read all images and labels from the database."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT image, label FROM images").fetchall()
    conn.close()
    X = np.array([preprocess(image) for image, _ in rows])
    y = np.array([label for _, label in rows])
    return X, y


def train():
    """Train the model, evaluate it, save it, and return the metrics."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    return metrics


if __name__ == "__main__":
    print(train())
