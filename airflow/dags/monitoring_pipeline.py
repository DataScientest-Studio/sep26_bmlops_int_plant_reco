"""Monitoring pipeline: check the latest predictions for drift."""
from datetime import datetime

import requests
from airflow.providers.standard.operators.python import (
    PythonOperator,
)
from airflow.sdk import DAG

API_URL = "http://api:8000"


def call_prediction_drift():
    """Ask the API to check the latest predictions for drift."""
    response = requests.post(
        f"{API_URL}/drift/predictions", params={"n": 30}, timeout=600
    )
    response.raise_for_status()
    print(response.json())


with DAG(
    dag_id="plant_reco_monitoring",
    schedule="0 12 * * *",  # daily at 12:00 UTC (training: midnight)
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    tags=["plant_reco"],
) as dag:
    PythonOperator(
        task_id="prediction_drift",
        python_callable=call_prediction_drift,
    )
