from fastapi import FastAPI
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pathlib import Path

app = FastAPI()

# Get absolute path to models folder
MODEL_PATH = Path(__file__).resolve().parent.parent / "models"

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

labels = ["billing", "account", "delivery", "technical"]


@app.get("/")
def home():
    return {"message": "Customer Support Ticket Classifier API"}


@app.get("/predict")
def predict(ticket: str):

    inputs = tokenizer(ticket, return_tensors="pt", truncation=True, padding=True)

    outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits).item()

    return {"ticket": ticket, "category": labels[prediction]}