from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from config.constant import classification_model_name, num_of_labels
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from config.constant import training_args
from src.data_preprocessing import Prepare_sentiment_data
import logging
from src.model_pusher import ModelPusher




class ModelTraining:
    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained(classification_model_name, num_labels=num_of_labels)
        
    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        return {"accuracy": acc, "f1":f1}
    
    def model_training(self, train_dataset, test_dataset):
        try:
            trainer = Trainer(
                model = self.model,
                args = training_args,
                train_dataset = train_dataset,
                eval_dataset = test_dataset,
                compute_metrics=self.compute_metrics
            )
            trainer.train()
            return trainer
        except Exception as e:
            logging.error(f"error occured during model training: {e}")

    def model_evaluation(self, trainer):
        result = trainer.evaluate()
        return result
