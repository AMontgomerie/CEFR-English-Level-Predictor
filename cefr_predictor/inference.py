import argparse
import numpy as np
from joblib import load

MIN_CONFIDENCE = 0.7
K = 2
LABELS = {
    0.0: "A1",
    0.5: "A1+",
    1.0: "A2",
    1.5: "A2+",
    2.0: "B1",
    2.5: "B1+",
    3.0: "B2",
    3.5: "B2+",
    4.0: "C1",
    4.5: "C1+",
    5.0: "C2",
    5.5: "C2+",
}


class Model:
    def __init__(self, model_path):
        self.model = load(model_path)

    def predict(self, data):
        probas = self.model.predict_proba(data)
        preds = [self._get_pred(p) for p in probas]
        probas = [self._label_probabilities(p) for p in probas]
        return preds, probas

    def predict_decode(self, data):
        preds, probas = self.predict(data)
        preds = [self.decode_label(p) for p in preds]
        return preds, probas

    def _get_pred(self, probabilities):
        if probabilities.max() < MIN_CONFIDENCE:
            return np.mean(probabilities.argsort()[-K:])
        else:
            return probabilities.argmax()

    def decode_label(self, encoded_label):
        return LABELS[encoded_label]

    def _label_probabilities(self, probas):
        labels = ["A1", "A2", "B1", "B2", "C1", "C2"]
        return {label: float(proba) for label, proba in zip(labels, probas)}


def parse_text_files():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text_files", nargs="+", default=[])
    args = parser.parse_args()

    texts = []
    for path in args.text_files:
        with open(path, "r") as f:
            texts.append(f.read())
    return texts


if __name__ == "__main__":
    texts = parse_text_files()
    if len(texts) == 0:
        raise Exception("Specify one or more documents to evaluate.")

    model = Model("cefr_predictor/models/xgboost.joblib")
    preds, probas = model.predict_decode(texts)

    results = []
    for text, pred, proba in zip(texts, preds, probas):
        row = {"text": text, "level": pred, "scores": proba}
        results.append(row)

    print(results)
