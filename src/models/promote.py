"""Compare a new model version with the production model and move the alias if it is better."""
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "plant_reco"
ALIAS = "production"
METRIC = "f1_macro"


def promote_if_better(new_version):
    """Give the new version the alias 'production' if its macro-F1 is higher."""
    client = MlflowClient()
    new = client.get_model_version(MODEL_NAME, new_version)
    new_score = client.get_run(new.run_id).data.metrics[METRIC]

    try:
        prod = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        prod_version = prod.version
        prod_score = client.get_run(prod.run_id).data.metrics[METRIC]
    except MlflowException:
        prod_version, prod_score = None, None

    promoted = prod_score is None or new_score > prod_score
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, new_version)

    return {
        "new_version": int(new_version),
        "new_f1_macro": new_score,
        "production_version_before": int(prod_version) if prod_version else None,
        "production_f1_macro_before": prod_score,
        "promoted": promoted,
    }
