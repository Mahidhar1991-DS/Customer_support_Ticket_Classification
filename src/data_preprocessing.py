import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def load_data(path):
    df = pd.read_csv(path)
    df["ticket_text"] = df["ticket_text"].apply(clean_text)
    return df