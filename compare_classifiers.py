import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from skopt import BayesSearchCV
import pickle

RANDOM_SEED = 0


def compare_models():
    """Compares several classifiers by performing Beyesian optimization on each one and
    then ranking the results.
    """

    model_configs = [
        {
            "name": "Random Forest",
            "model": RandomForestClassifier(random_state=RANDOM_SEED),
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
            "model": SVC(random_state=RANDOM_SEED),
            "params": {"C": (0.1, 10), "gamma": (0.0001, 0.1)},
        },
        {
            "name": "XGBoost",
            "model": xgb.XGBClassifier(
                objective="multi:softprob", random_state=RANDOM_SEED
            ),
            "params": {
                "learning_rate": (0.01, 0.5),
                "max_depth": (1, 10),
                "subsample": (0.8, 1.0),
                "colsample_bytree": (0.8, 1.0),
                "gamma": (0, 5),
                "n_estimators": (10, 500),
            },
        },
    ]

    train, test = load_data()

    results = []
    for model in model_configs:
        best_result = hyperparam_search(model, train, test)
        results.append(best_result)

    ranking = rank_results(results)
    best_model = ranking[0]

    save(best_model)


def load_data():
    data = pd.read_csv("data/CEFR/preprocessed_cefr_leveled_texts.csv")
    train, test = train_test_split(
        data, test_size=0.2, random_state=RANDOM_SEED, stratify=data.label
    )
    return train, test


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
    print("\n")
    return {"name": model_config["name"], "model": opt.best_estimator_, "score": acc}


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


def save(model_config):
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
