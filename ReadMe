# Multilingual Sentiment Analysis Application with MLOps Pipeline

## Overview

This project implements a full-stack sentiment analysis application that combines a user-facing interface with a transformer-based natural language processing backend and an evolving machine learning pipeline.

The application is designed to process and analyze customer reviews across multiple languages, classify sentiment into predefined categories, and provide insights that can support business decision-making. In addition to the application layer, the project incorporates machine learning engineering practices such as structured data processing, model training, evaluation, and experiment tracking.

The system is being developed progressively toward a production-grade architecture, with a focus on reproducibility, modularity, and scalability.

---

## Key Features

* Interactive web interface built with Streamlit for real-time sentiment analysis
* Multilingual text processing using transformer-based models (BERT / DistilBERT)
* Data preprocessing pipeline including text cleaning, lemmatization, and stopword removal
* Sentiment label engineering for multi-class classification (negative, neutral, positive)
* Train-test data splitting with reproducibility considerations
* Tokenization using HuggingFace tokenizers
* Custom PyTorch Dataset for efficient model input handling
* Model training and evaluation using HuggingFace Trainer API
* Evaluation metrics including accuracy and weighted F1-score
* Experiment tracking and model logging using MLflow
* Remote experiment tracking and artifact storage using DagsHub

---

## System Architecture

### Application Layer

User Input (Streamlit Interface)
→ Text Preprocessing
→ Tokenization
→ Model Inference
→ Sentiment Prediction Output

### Machine Learning Pipeline (In Progress)

Data Ingestion
→ Data Cleaning and Transformation
→ Feature Engineering
→ Train-Test Split
→ Tokenization
→ Dataset Construction
→ Model Training
→ Model Evaluation
→ Experiment Tracking (MLflow + DagsHub)
→ Model Artifact Storage

---

## Project Structure

```
shopease_app/
│
├── src/
│   ├── data_preprocessing.py     # Data cleaning, splitting, tokenization
│   ├── data_cleaning.py          # Text preprocessing logic
│   ├── model_training.py         # Model training and evaluation
│   ├── model_pusher.py           # MLflow logging and model tracking
│
├── config/
│   ├── constant.py               # Paths, model names, training arguments
│
├── Data/
│   ├── processed_data/           # Saved datasets (train/test)
│
├── artifacts/                    # Model outputs and logs
│
├── app.py                        # Streamlit application entry point
│
├── requirements.txt
└── README.md
```

---

## Technology Stack

* Python
* Streamlit
* PyTorch
* HuggingFace Transformers
* Scikit-learn
* MLflow
* DagsHub

---

## Model Details

* Base Model: BERT / DistilBERT (multilingual support considered)
* Task: Multi-class sentiment classification
* Labels:

  * 0: Negative
  * 1: Neutral
  * 2: Positive

---

## Machine Learning Workflow

The machine learning workflow follows a structured sequence:

1. Raw data is loaded and cleaned using a custom preprocessing module
2. Text data is normalized through cleaning, lemmatization, and stopword removal
3. Labels are generated from rating values
4. The dataset is split into training and testing sets
5. Text is tokenized using a pretrained transformer tokenizer
6. Encoded data is wrapped into a PyTorch Dataset
7. A transformer-based classification model is trained using HuggingFace Trainer
8. Model performance is evaluated using accuracy and F1-score
9. Metrics and models are logged to MLflow for experiment tracking
10. Artifacts are stored and managed using DagsHub

---

## Experiment Tracking

MLflow is used to:

* Track training experiments
* Log evaluation metrics
* Store trained model artifacts
* Enable reproducibility of experiments

DagsHub is used as a remote backend for:

* Centralized experiment tracking
* Version control for models and data artifacts

---

## Installation

1. Clone the repository:

```
git clone <repository-url>
cd shopease_app
```

2. Create and activate a virtual environment:

```
conda create -n shop_env python=3.10
conda activate shop_env
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Application

To run the Streamlit interface:

```
streamlit run app.py
```

---

## Running Model Training

```
python -m src.model_training
```

---

## Running MLflow Tracking

For local tracking:

```
mlflow ui
```

For DagsHub tracking, ensure credentials and tracking URI are configured in the environment.

---

## Project Status

* Streamlit application implemented
* Core NLP pipeline implemented
* Model training and evaluation functional
* MLflow integration implemented
* Remote tracking with DagsHub configured
* Pipeline modularization and optimization ongoing

---

## Future Improvements

* Hyperparameter tuning and model optimization
* Deployment via REST API (FastAPI)
* Real-time inference pipeline
* Model monitoring and drift detection
* Automated pipeline orchestration

---

## Author

This project demonstrates the integration of natural language processing, machine learning pipelines, and MLOps practices within a real-world application context.
