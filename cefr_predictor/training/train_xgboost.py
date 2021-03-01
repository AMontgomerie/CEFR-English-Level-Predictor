import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, FunctionTransformer
from cefr_predictor.preprocessing import generate_features

RANDOM_SEED = 0

label_encoder = None


def train():
    X_train, y_train = load_data("data/train.csv")
    X_test, y_test = load_data("data/test.csv")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print(pipeline.score(X_test, y_test))
    save_model(pipeline)


def build_pipeline():
    return Pipeline(
        steps=[
            ("generate features", FunctionTransformer(generate_features)),
            ("scale features", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    random_state=RANDOM_SEED,
                    use_label_encoder=False,
                ),
            ),
        ],
        verbose=True,
    )


def load_data(path_to_data):
    data = pd.read_csv(path_to_data)
    X = data.text.tolist()
    y = encode_labels(data.label)
    return X, y


def encode_labels(labels):
    global label_encoder
    if not label_encoder:
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
    return label_encoder.transform(labels)


def save_model(model):
    file_name = "cefr_predictor/models/xgboost.joblib"
    print(f"Saving {file_name}")
    dump(model, file_name)


if __name__ == "__main__":
    train()