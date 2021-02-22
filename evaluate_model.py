import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from util import load_data, load_model

LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def generate_confusion_matrix(model, test_set):
    X = test_set.drop("label", axis=1)
    y_true = test_set["label"]
    y_pred = model.predict(X)
    return confusion_matrix(y_true, y_pred)


def get_classification_report(model, test_set):
    X = test_set.drop("label", axis=1)
    y_true = test_set["label"]
    y_pred = model.predict(X)
    return classification_report(y_true, y_pred, target_names=LABELS)


def get_top_k_accuracy(model, test_set, k=1):
    X = test_set.drop("label", axis=1)
    y_true = test_set["label"]
    y_proba = model.predict_proba(X)
    return top_k_accuracy_score(y_true, y_proba, k)


def top_k_accuracy_score(y_true, y_proba, k=1):
    score = 0
    for proba, true in zip(y_proba, y_true):
        if true in proba.argsort()[-k:]:
            score += 1
    return score / len(y_true)


if __name__ == "__main__":
    _, test = load_data()
    model = load_model("models/cefr-xgboost.pickle")
    print(generate_confusion_matrix(model, test))
    print(get_classification_report(model, test))
    print(get_top_k_accuracy(model, test, k=2))
