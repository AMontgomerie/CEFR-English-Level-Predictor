from textstat import textstat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, LabelEncoder
from sklearn.metrics import accuracy_score

LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

METRICS = [
    textstat.flesch_reading_ease,
    textstat.smog_index,
    textstat.flesch_kincaid_grade,
    textstat.coleman_liau_index,
    textstat.automated_readability_index,
    textstat.dale_chall_readability_score,
    textstat.difficult_words,
    textstat.linsear_write_formula,
    textstat.gunning_fog,
    textstat.text_standard,
]


class Predictor:
    def __init__(self, prediction_function):
        self.predict_func = prediction_function
        self.scaler = MinMaxScaler(feature_range=(0, len(LABELS) - 1))

    def predict(self, X):
        output = X.apply(self._predict_text)
        output = pd.DataFrame(output)
        scaled_outputs = pd.DataFrame(self.scaler.fit_transform(output))
        return [round(p) for p in scaled_outputs[0]]

    def get_name(self):
        return self.predict_func.__name__

    def _predict_text(self, text):
        if self.get_name() == "text_standard":
            return self.predict_func(text, float_output=True)
        else:
            return self.predict_func(text)


def load_data():
    test = pd.read_csv("data/test.csv")
    X = test.text
    y = test.label
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    return X, y


def calculate_metrics(X, y):
    for metric in METRICS:
        predictor = Predictor(metric)
        preds = predictor.predict(X)
        score = accuracy_score(y, preds)
        print(f"{predictor.get_name()}: {score}")


if __name__ == "__main__":
    X, y = load_data()
    print(f"baseline random: {1 / len(LABELS)}")
    calculate_metrics(X, y)