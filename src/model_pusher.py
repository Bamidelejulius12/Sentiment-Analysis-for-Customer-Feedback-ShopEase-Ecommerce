import mlflow
import mlflow.pytorch
import logging
import dagshub

from utils.model_utils import get_best_f1
from config.constant import training_args, classification_model_name


class ModelPusher:
    def __init__(self, experiment_name = "sentiment-analysis"):
        try:
            dagshub.init(
                repo_owner='babatundejulius911', 
                repo_name='Sentiment-Analysis-for-Customer-Feedback-ShopEase-Ecommerce', 
                mlflow=True
            )
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
        except Exception as e:
            logging.error(f"error while initializting mlflow: {e}")

    def updated_model_pusher(self, trainer, metrics):
        try:
            new_f1 = metrics["eval_f1"]
            old_f1 = get_best_f1(self.experiment_name)

            print(f"Previous model F1: {old_f1}")
            print(f"New model F1: {new_f1}")

            if old_f1 is None or new_f1 > old_f1:
                with mlflow.start_run():

                    # Log single metrics correctly
                    mlflow.log_metric("accuracy", metrics["eval_accuracy"])
                    mlflow.log_metric("f1", new_f1)

                    # Log parameters
                    mlflow.log_param("model_name", classification_model_name)
                    mlflow.log_param("epochs", training_args.num_train_epochs)
                    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
                    mlflow.log_param("learning_rate", training_args.learning_rate)

                    # Log the model
                    mlflow.pytorch.log_model(
                        pytorch_model= trainer.model,
                        artifact_path="model",
                        registered_model_name=classification_model_name
                    )

                print("New model pushed to MLflow with better performance.")
            else:
                print("Model not pushed, as the previous model still has better performance.")

        except Exception as e:
            logging.error(f"Model pushing error: {e}")