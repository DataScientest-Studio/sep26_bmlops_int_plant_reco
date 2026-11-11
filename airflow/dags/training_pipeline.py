"""Training pipeline: ETL, drift check, then train and compare."""
from datetime import datetime

import requests
from airflow.providers.standard.operators.python import (
    PythonOperator,
)
from airflow.sdk import DAG

API_URL = "http://api:8000"


def call_etl():
    """Ask the API to add the next batch of new images."""
    response = requests.post(
        f"{API_URL}/etl", params={"per_class": 20}, timeout=600
    )
    response.raise_for_status()
    print(response.json())


def call_image_drift():
    """Ask the API to check the new images for drift (Evidently)."""
    response = requests.post(f"{API_URL}/drift/images", timeout=600)
    response.raise_for_status()
    print(response.json())


def call_train_evaluate():
    """Ask the API to train, evaluate, register and compare a model."""
    response = requests.post(
        f"{API_URL}/train_evaluate", timeout=1800
    )
    response.raise_for_status()
    print(response.json())


with DAG(
    dag_id="plant_reco_training",
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["plant_reco"],
) as dag:
    etl = PythonOperator(task_id="etl", python_callable=call_etl)
    image_drift = PythonOperator(
        task_id="image_drift", python_callable=call_image_drift
    )
    train_evaluate = PythonOperator(
        task_id="train_evaluate", python_callable=call_train_evaluate
    )

    etl >> image_drift >> train_evaluate
