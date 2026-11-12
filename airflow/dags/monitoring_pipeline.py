"""Monitoring pipeline: check the latest predictions for drift,
and start the training pipeline when drift is found."""
from datetime import datetime

import requests
from airflow.providers.standard.operators.python import (
    ShortCircuitOperator,
)
from airflow.providers.standard.operators.trigger_dagrun import (
    TriggerDagRunOperator,
)
from airflow.sdk import DAG

API_URL = "http://api:8000"


def call_prediction_drift():
    """Check the latest predictions; True means drift was found."""
    response = requests.post(
        f"{API_URL}/drift/predictions", params={"n": 30}, timeout=600
    )
    response.raise_for_status()
    result = response.json()
    print(result)
    # True: the next task runs. False or None: it is skipped.
    return bool(result["drift"])


with DAG(
    dag_id="plant_reco_monitoring",
    schedule="0 12 * * *",  # daily at 12:00 UTC (training: midnight)
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    tags=["plant_reco"],
) as dag:
    prediction_drift = ShortCircuitOperator(
        task_id="prediction_drift",
        python_callable=call_prediction_drift,
    )
    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="plant_reco_training",
    )

    prediction_drift >> trigger_training
