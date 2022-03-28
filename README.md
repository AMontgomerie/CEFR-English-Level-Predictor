# CEFR-English-Level-Predictor

NLP system for predicting the reading difficulty level of a text in terms of its CEFR level.

## Try the Streamlit app

https://share.streamlit.io/amontgomerie/cefr-english-level-predictor/main/CEFR_Predictor.py

## Run the Streamlit app as a Docker container

```
git clone https://github.com/AMontgomerie/CEFR-English-Level-Predictor
cd CEFR-English-Level-Predictor
docker build -t cefr-predictor .
docker run -p 8080:8080 -d cefr-predictor
```

## Install as a package and run locally

```
git clone https://github.com/AMontgomerie/CEFR-English-Level-Predictor
cd CEFR-English-Level-Predictor
python -m pip install -e .
```

### Run as Streamlit app locally

```
streamlit run CEFR_Predictor.py
```

Then open `http://localhost:8501/`

### Run as API locally

```
uvicorn api:app --reload
```

Then open `http://127.0.0.1:8000/docs`
