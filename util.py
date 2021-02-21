import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

RANDOM_SEED = 0


def load_data():
    data = pd.read_csv("data/preprocessed_cefr_leveled_texts.csv")
    train, test = train_test_split(
        data, test_size=0.2, random_state=RANDOM_SEED, stratify=data.label
    )
    return train, test


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


def load_model(path_to_model):
    return pickle.load(open(path_to_model, "rb"))