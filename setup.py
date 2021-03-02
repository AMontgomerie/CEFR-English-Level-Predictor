from setuptools import setup, find_packages

setup(
    name="EnglishCEFRPredictor",
    version="1.0.0",
    author_email="adam.montgomerie971@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy==1.20.1",
        "pandas==1.2.2",
        "textstat==0.7.0",
        "xgboost==1.3.3",
        "scikit-learn==0.24.1",
        "scikit-optimize==0.8.1",
        "streamlit==0.77.0",
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz",
    ],
)