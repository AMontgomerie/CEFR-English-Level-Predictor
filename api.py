from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from cefr_predictor.inference import Model

app = FastAPI()

model = Model("cefr_predictor/models/xgboost.joblib")


class TextList(BaseModel):
    texts: List[str] = []


@app.get("/")
def root():
    return {"message": "Nothing to see here."}


@app.post("/predict")
def predict(textlist: TextList):
    preds, probas = model.predict_decode(textlist.texts)

    response = []
    for text, pred, proba in zip(textlist.texts, preds, probas):
        row = {"text": text, "level": pred, "scores": proba}
        response.append(row)

    return response
