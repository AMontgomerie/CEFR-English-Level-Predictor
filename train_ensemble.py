import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from util import load_data, save_model
from hyperparam_search import hyperparam_search, get_model_configs

RANDOM_SEED = 0


def train_ensemble():
    train, test = load_data()
    X_train = train.drop("label", axis=1)
    y_train = train.label
    X_test = test.drop("label", axis=1)
    y_test = test.label

    models = get_best_model_configs(train, test)

    for voting in ["hard", "soft"]:
        for weights in [[1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1]]:

            ensemble = VotingClassifier(models, voting=voting, weights=weights)
            ensemble.fit(X_train, y_train)
            print(weights, voting)
            print(accuracy_score(y_test, ensemble.predict(X_test)))


def get_best_model_configs(train, test):
    model_configs = get_model_configs()

    results = [
        hyperparam_search(model, train, test)
        for model in model_configs
        if model["name"] != "SVC"
    ]

    models = [
        (
            model["name"],
            model["class"](**model["params"], random_state=RANDOM_SEED),
        )
        for model in results
    ]

    return models


if __name__ == "__main__":
    train_ensemble()
