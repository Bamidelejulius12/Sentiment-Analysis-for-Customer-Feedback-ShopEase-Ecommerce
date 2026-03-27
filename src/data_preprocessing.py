import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertTokenizer
import logging
import torch
import os
from config.constant import train_Data, test_Data, Cleaned_Data, Input_Data
from src.data_cleaning import clean_data

class data_processor:
    def __init__(self):
        self.data = clean_data(pd.read_csv(Input_Data))

    def split_data(self):
        try:
            X = self.data["final_text"].astype(str)
            y = self.data['label']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("error occurred while splitting data")

    
class TokenizerWrapper:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def encode(self, texts):
        return self.tokenizer(
            texts.to_list(),
            truncation = True,
            padding=True,
            max_length = 128
        )


class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        # Ensure labels are a list without pandas index issues
        if hasattr(labels, 'tolist'):
            self.labels = labels.tolist()  # Convert pandas Series to list
        elif hasattr(labels, '__iter__') and not isinstance(labels, (list, tuple)):
            self.labels = list(labels)
        else:
            self.labels = labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item



def Prepare_sentiment_data():
    try:
        processor = data_processor()

        X_train, X_test, y_train, y_test = processor.split_data()
        if hasattr(y_train, 'tolist'):
            y_train = y_train.tolist()
        if hasattr(y_test, 'tolist'):
            y_test = y_test.tolist()
        tokenizer = TokenizerWrapper()
        train_encodings = tokenizer.encode(X_train)
        test_encodings = tokenizer.encode(X_test)

        train_dataset = SentimentDataset(train_encodings, y_train)
        test_dataset = SentimentDataset(test_encodings, y_test)
        os.makedirs(os.path.dirname(train_Data), exist_ok=True)
        os.makedirs(os.path.dirname(test_Data), exist_ok=True)
        torch.save(train_dataset, train_Data)
        torch.save(test_dataset, test_Data)
        logging.info("pipeline completed successfully")
        return train_dataset, test_dataset
    except Exception as e:
        logging.error(f"error occurred during data set processing {e}")

    






        

            
            




