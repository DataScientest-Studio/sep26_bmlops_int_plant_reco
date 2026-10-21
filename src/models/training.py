"""Train a baseline model on the images in the database and track it in MLflow."""
import io

import mlflow
import mlflow.sklearn
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.models.promote import MODEL_NAME, promote_if_better

IMG_SIZE = 32
SEED = 42
N_ESTIMATORS = 100
EXPERIMENT_NAME = "plant_reco"


def preprocess(image_bytes):
    """Image bytes -> flat vector of 32*32*3 numbers between 0 and 1."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32).flatten() / 255.0


def load_data():
    """Read all images and labels from PostgreSQL in stable order."""
    from src.data.db import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT image, label FROM images ORDER BY id")
            rows = cur.fetchall()

    X = np.array([preprocess(row["image"]) for row in rows])
    y = np.array([row["label"] for row in rows])
    return X, y


def train():
    """Train, evaluate, log the run in MLflow, register the model and compare it."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, n_jobs=-1, random_state=SEED)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        }

        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "img_size": IMG_SIZE,
            "seed": SEED,
            "n_train": len(y_train),
            "n_test": len(y_test),
        })
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            model, name="model", registered_model_name=MODEL_NAME
        )

    comparison = promote_if_better(model_info.registered_model_version)
    return {
        **metrics,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "run_id": run.info.run_id,
        **comparison,
    }


if __name__ == "__main__":
    print(train())
