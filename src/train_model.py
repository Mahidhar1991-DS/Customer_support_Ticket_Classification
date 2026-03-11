import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import json

df = pd.read_csv("data/sample_tickets.csv")

labels = list(df["category"].unique())
label_dict = {label: i for i, label in enumerate(labels)}

# SAVE LABEL MAP HERE
json.dump(label_dict, open("models/label_map.json","w"))

df["label"] = df["category"].map(label_dict)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["ticket_text"], df["label"], test_size=0.2
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(labels)
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=8,
    per_device_train_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

model.save_pretrained("models/")
tokenizer.save_pretrained("models/")