# main/app.py
import logging
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import io
import numpy as np
from pipeline.prediction import predict_sentiment
from pipeline.training import Train_model

# ---------- CONFIG ----------
logging.basicConfig(level=logging.INFO)

# ---------- FASTAPI APP ----------
app = FastAPI(title="ShopEase Sentiment API")

# ---------- Pydantic Schemas ----------
class TextRequest(BaseModel):
    text: str

# ---------- LOAD MODEL ON STARTUP ----------
predictor = predict_sentiment()
logging.info("Model loaded successfully")

# ---------- PREDICTION ROUTE ----------
@app.post("/predict")
def predict_text(request: TextRequest):
    result = predictor.predict(request.text)
    top_label = max(result, key=lambda x: x['score'])
    return {"label": top_label['label'], "confidence": float(top_label['score'])}

# ---------- BATCH PREDICTION ROUTE ----------
@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Upload CSV with 'review' column, get back predictions as table"""
    # Read CSV
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    # Check for review column
    if 'review' not in df.columns:
        return {"error": "CSV must have a 'review' column"}
    
    # Predict each review
    results_list = []
    for idx, row in df.iterrows():
        try:
            review = str(row['review'])
            
            # Call predict and see what happens
            result = predictor.predict(review)
            
            if result is None or len(result) == 0:
                raise ValueError("Empty result from model")
            
            top_label = max(result, key=lambda x: x['score'])
            
            result_row = row.to_dict()
            result_row['sentiment_label'] = top_label['label']
            result_row['sentiment_confidence'] = float(top_label['score'])
            results_list.append(result_row)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logging.error(f"Error for review {idx}: {error_details}")
            
            result_row = row.to_dict()
            result_row['sentiment_label'] = f"ERROR: {str(e)[:50]}"
            result_row['sentiment_confidence'] = 0.0
            results_list.append(result_row)
    
    return results_list

@app.get("/train")
def train_model():
    try:
        Train_model()
    except Exception as e:
        logging.error(f"Training error: {e}")
        return {"error": f"Training failed: {str(e)}"}