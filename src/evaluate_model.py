from sklearn.metrics import classification_report
import pandas as pd

def evaluate(true_labels, predictions):
    print(classification_report(true_labels, predictions))