import os
from transformers import Trainer, TrainingArguments

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Input_Data = os.path.join(BASE_DIR, 'Data', 'testing_data.csv')
Cleaned_Data = os.path.join(BASE_DIR, 'Data', 'cleaned_data', 'clean_data.csv')
train_Data = os.path.join(BASE_DIR, 'Data', 'processed_data', 'train_data.pt')
test_Data = os.path.join(BASE_DIR, 'Data', 'processed_data', 'test_data.pt')
model_name = "distilbert-base-multilingual-cased"
classification_model_name = "bert-base-uncased"
num_of_labels = 3

training_args = TrainingArguments(
    output_dir = "./results",
    num_train_epochs = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 32,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    logging_dir = "./logs",
    logging_steps = 50,
    save_total_limit = 1,
    load_best_model_at_end = True,
    metric_for_best_model = "accuracy"

)