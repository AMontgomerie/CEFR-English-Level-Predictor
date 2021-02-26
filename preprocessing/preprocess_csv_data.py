import pandas as pd
from preprocessing import preprocess
from sklearn.model_selection import train_test_split

RANDOM_SEED = 0


if __name__ == "__main__":
    print("Starting preprocessing...")

    data = pd.read_csv("data/cefr_leveled_texts.csv", encoding="utf-8")
    train, test = train_test_split(
        data, test_size=0.2, random_state=RANDOM_SEED, stratify=data.label
    )

    print("Preparing Training Data.")
    prepped_train = preprocess(train)
    prepped_train.to_csv("data/train.csv")

    print("Preparing Test Data.")
    prepped_test = preprocess(test)
    prepped_test.to_csv("data/test.csv")

    print("Finished preprocessing.")