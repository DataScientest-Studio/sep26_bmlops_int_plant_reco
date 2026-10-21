"""Training pipeline: add new images (ETL), then train, evaluate and compare the model."""
from datetime import datetime

import requests
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import DAG

API_URL = "http://api:8000"


def call_etl():
    """Ask the API to add the next batch of new images to the database."""
    response = requests.post(f"{API_URL}/etl", params={"per_class": 20}, timeout=600)
    response.raise_for_status()
    print(response.json())


def call_train_evaluate():
    """Ask the API to train, evaluate, register and compare a new model."""
    response = requests.post(f"{API_URL}/train_evaluate", timeout=1800)
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
    train_evaluate = PythonOperator(task_id="train_evaluate", python_callable=call_train_evaluate)

    etl >> train_evaluate
