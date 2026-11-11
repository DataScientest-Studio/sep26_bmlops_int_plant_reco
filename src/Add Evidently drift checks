"""Data drift checks with Evidently, tracked in MLflow.

Both checks compare simple image features with the first load
of images (the historical data):
- images: the newest batch added by the ETL
- predictions: the latest images sent to POST /predict
"""
import mlflow
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.data.db import get_connection
from src.features.build_features import image_features

FEATURES = [
    "brightness", "contrast", "mean_r", "mean_g", "mean_b",
    "sharpness",
]
EXPERIMENT_NAME = "plant_reco_drift"
# A feature has drifted if the K-S test gives a p-value below this
P_VALUE = 0.001


def batch_features(which):
    """Features of the first ("MIN") or newest ("MAX") batch."""
    query = (
        "SELECT image FROM images "
        f"WHERE added_at = (SELECT {which}(added_at) FROM images)"
    )
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    features = [image_features(row["image"]) for row in rows]
    return pd.DataFrame(features)


def prediction_features(n):
    """Features of the n latest predictions."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT brightness, contrast, mean_r, mean_g, mean_b,
                       sharpness
                FROM predictions ORDER BY id DESC LIMIT %s
                """,
                (n,),
            )
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=FEATURES)


def compare(name, reference, current):
    """Run Evidently, log the result and the report in MLflow."""
    definition = DataDefinition(numerical_columns=FEATURES)
    report = Report([DataDriftPreset(method="ks", threshold=P_VALUE)])
    snapshot = report.run(
        current_data=Dataset.from_pandas(
            current[FEATURES], data_definition=definition
        ),
        reference_data=Dataset.from_pandas(
            reference[FEATURES], data_definition=definition
        ),
    )

    p_values = {}
    for metric in snapshot.dict()["metrics"][1:]:
        metric_id = metric["metric_id"]
        column = metric_id.split("column=")[1].split(",")[0]
        p_values[column] = float(metric["value"])
    drifted = [col for col, p in p_values.items() if p < P_VALUE]

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({
            "check": name,
            "test": "ks",
            "p_value_threshold": P_VALUE,
            "n_reference": len(reference),
            "n_current": len(current),
        })
        mlflow.log_metrics({
            "drift": int(bool(drifted)),
            "drifted_features": len(drifted),
            **{f"p_value_{col}": p for col, p in p_values.items()},
        })
        report_path = f"/tmp/drift_report_{name}.html"
        snapshot.save_html(report_path)
        mlflow.log_artifact(report_path)

    return {
        "check": name,
        "drift": bool(drifted),
        "drifted_features": drifted,
        "n_reference": len(reference),
        "n_current": len(current),
        "run_id": run.info.run_id,
    }


def check_image_drift():
    """Compare the newest batch of images with the first load."""
    first, newest = batch_features("MIN"), batch_features("MAX")
    return compare("images", first, newest)


def check_prediction_drift(n=30):
    """Compare the n latest predictions with the first load."""
    current = prediction_features(n)
    if len(current) < n:
        return {
            "check": "predictions",
            "drift": None,
            "message": f"only {len(current)} predictions, need {n}",
        }
    return compare("predictions", batch_features("MIN"), current)
