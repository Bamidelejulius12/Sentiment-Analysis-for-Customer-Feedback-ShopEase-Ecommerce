import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from config.constant import Input_Data, Cleaned_Data


def data_ingestion():
    try:
     data = pd.read_csv(Input_Data)
     logging.info(f"data has been successfully imported...")
     print(data.head())
     return data
    except Exception as e:
       logging.error(f"File to load the dataset ", e)


