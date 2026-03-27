from langdetect import detect
import spacy
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.data_ingestion import data_ingestion
import re
import sys
import logging
from config.constant import Cleaned_Data

sentiment_data = data_ingestion()
class DataCleaning:
    def __init__(self):
        self._ensure_nltk()
        
    def _load_nlp(self) -> spacy.language.Language:
        for model in ("en_core_web_sm", "xx_ent_wiki_sm"):
            try:
                nlp_model = spacy.load(model)
                logging.info("model successfully loaded")
                return nlp_model
            except OSError:
                continue
        nlp_fallback = spacy.blank("xx")
        return nlp_fallback
    
    def _ensure_nltk(self) -> None:
        try:
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords")
        
        try:
            word_tokenize("test")
        except LookupError:
            nltk.download("punkt")
        ## Newer NLTK versions split tokenize tables into 'punkt_tab'
        try:
            nltk.data.find("tokenizers/punkt_tab/english")
        except LookupError:
            try:
                nltk.download("punkt_tab")
            except Exception:
                pass

    def detect_language(text:str) -> str:
        try:
            return detect(text)
        except Exception:
            return "unknown"
        
    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def lemmatize(self, text: str) -> str:
        NLP = self._load_nlp()
        doc = NLP(text)
        return " ".join(token.lemma if token.lemma else token.text for token in doc)
    
    def remove_stopwords(self, text:str) -> str:
        tokens = word_tokenize(text)
        sw = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in sw]
        return " ".join(tokens)
    

def clean_data(data: pd.DataFrame):
    try:
        cleaner = DataCleaning()
        data['clean_text'] = data['review'].apply(cleaner.clean_text)
        data['lemma_text'] = data['clean_text'].apply(cleaner.lemmatize)
        data['final_text'] = data['lemma_text'].apply(cleaner.remove_stopwords)

        # Create the answer key for the AI
        data['label'] = data['rating'].apply(
            lambda r: 0 if r in (1, 2) else (1 if r == 3 else 2)
        )
        data = data[["review", "final_text", "label"]]
        
        data = data.drop_duplicates(subset="final_text")
        logging.info("data sucessfully cleaned")
        data.to_csv(Cleaned_Data)
        return data
    except Exception as e:
        logging.error(f"error occurred while cleaning data: {e}")


data = clean_data(sentiment_data)