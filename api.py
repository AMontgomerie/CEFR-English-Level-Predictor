from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from inference import Model
from preprocessing.preprocessing import preprocess_list

app = FastAPI()

model = Model("models/cefr-xgboost.pickle")


class Texts(BaseModel):
    texts: List[str] = []


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict")
def predict(texts: Texts):
    prepped_data = preprocess_list(texts.texts)
    preds, probas = model.predict_decode(prepped_data)

    response = []
    for text, pred, proba in zip(texts.texts, preds, probas):
        row = {"text": text, "level": pred, "scores": proba}
        response.append(row)

    return response
