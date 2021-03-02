import streamlit as st

from cefr_predictor.inference import Model

model = None


def load_model():
    return Model("cefr_predictor/models/xgboost.joblib")


def app():
    st.write("# English CEFR Level Predictor")

    text = st.text_area("Type or paste a text here:", height=200)

    if text:
        output = predict([text])
        display_results(output)

    st.write(
        "For more information: [amontgomerie.github.io](https://amontgomerie.github.io)"
    )


def predict(textlist):
    preds, probas = model.predict_decode(textlist)

    return {"level": preds[0], "scores": probas[0]}


def display_results(results):
    st.write(f"## Predicted CEFR level: __{results['level']}__")

    st.write("Text score per level:")
    st.write(results["scores"])


if __name__ == "__main__":
    model = load_model()
    app()