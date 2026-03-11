# BERT NLP Customer Support Ticket Classification

## Overview

This project builds a Natural Language Processing model to automatically classify customer support tickets into issue categories such as billing, account issues, delivery problems, and technical errors.

The model uses a BERT transformer from HuggingFace to understand ticket text and predict the correct category.

## Features

- NLP text preprocessing
- BERT-based classification
- Model evaluation with precision, recall, and F1-score
- Prediction script
- Optional API for real-time classification

## Technologies

- Python
- PyTorch
- HuggingFace Transformers
- Pandas
- Scikit-learn
- FastAPI

## Dataset Example

| ticket_text                       | category  |
| --------------------------------- | --------- |
| Payment failed but money deducted | billing   |
| Cannot login to my account        | account   |
| App crashes on startup            | technical |


