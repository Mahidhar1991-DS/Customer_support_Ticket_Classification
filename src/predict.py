from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("models/")
tokenizer = BertTokenizer.from_pretrained("models/")

labels = ["billing", "account", "delivery", "technical"]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    predicted_class_id = torch.argmax(outputs.logits).item()
    return labels[predicted_class_id]

ticket = input("Enter support ticket: ")

print("Predicted Category:", predict(ticket))