from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from util import load_data, load_model

LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def generate_confusion_matrix(model, test_set):
    X = test_set.drop("label", axis=1)
    y_true = test_set["label"]
    y_pred = model.predict(X)
    matrix = confusion_matrix(y_true, y_pred)
    print(accuracy_score(y_true, y_pred))
    print(matrix)


def get_classification_report(model, test_set):
    X = test_set.drop("label", axis=1)
    y_true = test_set["label"]
    y_pred = model.predict(X)
    report = classification_report(y_true, y_pred, target_names=LABELS)
    print(report)


if __name__ == "__main__":
    _, test = load_data()
    model = load_model("models/cefr-xgboost.pickle")
    generate_confusion_matrix(model, test)
    get_classification_report(model, test)
