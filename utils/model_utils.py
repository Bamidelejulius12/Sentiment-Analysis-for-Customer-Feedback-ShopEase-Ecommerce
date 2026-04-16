import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import logging
import os



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

def load_registered_model(model_name="distilbert-multilingual-sentiment"):
    # Local testing initialization of mlflow
    # dagshub.init(
    #     repo_owner='babatundejulius911',
    #     repo_name='Sentiment-Analysis-for-Customer-Feedback-ShopEase-Ecommerce',
    #     mlflow=True
    # )

    # Remote / Production access to mflow dagshub
    dagshub_token = os.getenv("Shop_env_DAGSHUB_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("Shop_env_DAGSHUB_TOKEN environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "babatundejulius911"
    repo_name = "Sentiment-Analysis-for-Customer-Feedback-ShopEase-Ecommerce"
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
  
    model_uri = f"models:/{model_name}/latest"

    sentiment_pipeline = mlflow.transformers.load_model(model_uri)

    return sentiment_pipeline