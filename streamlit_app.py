import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path
import json 

# Load model
MODEL_PATH = Path("models")

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

label_map = json.load(open("models/label_map.json"))
labels = {v:k for k,v in label_map.items()}

st.set_page_config(page_title="AI Support Ticket Classifier")

st.title("🤖 Customer Support Ticket Classifier")

st.write(
"This AI model classifies customer support tickets into categories like billing, account, delivery, or technical issues."
)

ticket = st.text_area("Enter Support Ticket")

if st.button("Predict Category"):

    if ticket.strip() == "":
        st.warning("Please enter a support ticket.")
    else:

        inputs = tokenizer(ticket, return_tensors="pt", truncation=True, padding=True)

        outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits).item()

        category = labels[prediction]

        st.success(f"Predicted Category: **{category.upper()}**")