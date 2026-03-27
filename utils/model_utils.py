import mlflow
from mlflow.tracking import MlflowClient

def get_best_model(experiment_name = "sentiment-analysis"):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    
    runs = client.search_runs([experiment.experiment_id])
    if not runs:
        return None
    best_model = sorted(
        runs,
        key = lambda x:x.data.metrics.get("f1", 0),
        reverse= True
    )[0]

    return best_model

def get_best_f1(experiment_name="sentiment-analysis"):
    best_run = get_best_model(experiment_name)
    if best_run is None:
        return None
    return best_run.data.metrics.get("f1", 0)

def load_best_model(experiment_name = "sentiment-analysis"):
    best_run = get_best_model(experiment_name)
    if best_run is None:
        return None
    
    model_uri = f"runs:/{best_run.info.run_id}/model"
    return mlflow.pytorch.load_model(model_uri)
