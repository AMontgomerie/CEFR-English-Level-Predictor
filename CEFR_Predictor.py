import streamlit as st
from io import StringIO

from cefr_predictor.inference import Model

model = None


def load_model():
    return Model("cefr_predictor/models/xgboost.joblib")


def app():
    st.write("# English CEFR Level Predictor")

    textbox_text = st.text_area("Type or paste a text here:", height=200)

    uploaded_files = st.file_uploader(
        "Or choose one or more text files to upload",
        type=["txt"],
        accept_multiple_files=True,
    )

    if st.button("Predict") or textbox_text:
        input_texts = collect_inputs(textbox_text, uploaded_files)

        if input_texts:
            output = model.predict_decode(input_texts)
            display_results(input_texts, output)

        else:
            st.write("Input one or more texts to generate a prediction.")

    st.write(
        "For more information: [amontgomerie.github.io](https://amontgomerie.github.io)"
    )


def collect_inputs(textbox_text, uploaded_files):
    inputs = []

    if textbox_text:
        inputs.append(textbox_text)

    if uploaded_files:
        for upload in uploaded_files:
            stringio = StringIO(upload.getvalue().decode("utf-8"))
            text = stringio.read()
            inputs.append(text)

    return inputs


def display_results(texts, output):
    levels, scores = output

    for i, (text, level, score) in enumerate(zip(texts, levels, scores)):
        st.write(f"### Text {i+1}:")
        st.write(f"_{text}_")
        st.write(f"### Predicted CEFR level: __{level}__")
        st.write("### Text score per level:")
        st.write(score)


if __name__ == "__main__":
    model = load_model()
    app()