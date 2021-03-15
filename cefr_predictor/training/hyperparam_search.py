import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from skopt import BayesSearchCV
import pickle

RANDOM_SEED = 0


def compare_models():
    """Compares several classifiers by performing Beyesian optimization on each one and
    then ranking the results.
    """
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    results = []

    for configs in get_model_configs():
        best_result = hyperparam_search(configs, train, test)
        save_model(best_result)
        results.append(best_result)

    rank_results(results)


def hyperparam_search(model_config, train, test):
    """Perform hyperparameter search using Bayesian optimization on a given model and
    dataset.

    Args:
        model_config (dict): the model and the parameter ranges to search in. Format:
        {
            "name": str,
            "model": sklearn.base.BaseEstimator,
            "params": dict
        }
        train (pandas.DataFrame): training data
        test (pandas.DataFrame): test data
    """
    X_train = train.drop("label", axis=1)
    y_train = train.label

    X_test = test.drop("label", axis=1)
    y_test = test.label

    opt = BayesSearchCV(
        model_config["model"],
        model_config["params"],
        n_jobs=4,
        cv=5,
        random_state=RANDOM_SEED,
    )
    opt.fit(X_train, y_train)
    acc = opt.score(X_test, y_test)

    print(f"{model_config['name']} results:")
    print(f"Best validation accuracy: {opt.best_score_}")
    print(f"Test set accuracy: {acc}")
    print(f"Best parameters:")

    for param, value in opt.best_params_.items():
        print(f"- {param}: {value}")

    return {
        "name": model_config["name"],
        "class": model_config["class"],
        "model": opt.best_estimator_,
        "params": opt.best_params_,
        "score": acc,
    }


def get_model_configs():
    return [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(random_state=RANDOM_SEED),
            "class": LogisticRegression,
            "params": {
                "C": (0.1, 10),
                "fit_intercept": [True, False],
                "max_iter": (100, 1000),
            },
        },
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(random_state=RANDOM_SEED),
            "class": RandomForestClassifier,
            "params": {
                "bootstrap": [True, False],
                "max_depth": (10, 100),
                "max_features": ["auto", "sqrt"],
                "min_samples_leaf": (1, 5),
                "min_samples_split": (2, 10),
                "n_estimators": (100, 500),
            },
        },
        {
            "name": "SVC",
            "model": SVC(random_state=RANDOM_SEED, probability=True),
            "class": SVC,
            "params": {"C": (0.1, 10), "gamma": (0.0001, 0.1)},
        },
        {
            "name": "XGBoost",
            "model": XGBClassifier(
                objective="multi:softprob", random_state=RANDOM_SEED
            ),
            "class": XGBClassifier,
            "params": {
                "learning_rate": (0.01, 0.1),
                "max_depth": (1, 10),
                "subsample": (0.8, 1.0),
                "colsample_bytree": (0.8, 1.0),
                "gamma": (1, 5),
                "n_estimators": (50, 500),
            },
        },
    ]


def rank_results(results):
    """Ranks the results of the hyperparam search, prints the ranks and returns them.

    Args:
        results (list): list of model configs and their scores.

    Returns:
        list: the results list reordered by performance.
    """
    ranking = sorted(results, key=lambda k: k["score"], reverse=True)
    for i, rank in enumerate(ranking):
        print(f"{i+1}. {rank['name']}: {rank['score']}")
    print("\n")
    return ranking


def save_model(model_config):
    """Saves a model.

    Args:
        model_config: a dict with a "name" key and a "model" key with the trained model
        object.
    """
    model_name = model_config["name"].lower().replace(" ", "_")
    file_name = f"models/cefr-{model_name}.pickle"
    print(f"Saving {file_name}")
    pickle.dump(model_config["model"], open(file_name, "wb"))


if __name__ == "__main__":
    compare_models()