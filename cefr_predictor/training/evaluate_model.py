import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def get_data():
    test = pd.read_csv("data/test.csv")
    X = test["text"]
    y = test["label"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y


def get_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    matrix_df = pd.DataFrame(matrix, columns=LABELS)
    matrix_df["label"] = LABELS
    return matrix_df[["label"] + LABELS]


def get_top_k_accuracy(model, X, y_true, k=1):
    y_proba = model.predict_proba(X)
    return top_k_accuracy_score(y_true, y_proba, k)


def top_k_accuracy_score(y_true, y_proba, k=1):
    score = 0
    for proba, true in zip(y_proba, y_true):
        if true in proba.argsort()[-k:]:
            score += 1
    return score / len(y_true)


if __name__ == "__main__":
    X, y_true = get_data()
    model = load("cefr_predictor/models/xgboost.joblib")
    y_pred = model.predict(X)
    print(get_confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=LABELS))
    print(get_top_k_accuracy(model, X, y_true, k=2))
