# src/models/dataloader.py
import os
import pandas as pd
from src.preprocessing.text_cleaner import clean_text

def load_csv_dataset(path: str, text_col='text', label_col='label'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    return df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
