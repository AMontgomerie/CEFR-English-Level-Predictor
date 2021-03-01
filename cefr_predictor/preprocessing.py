from textstat import textstat
import re
import os
import en_core_web_sm
import numpy as np
import pandas as pd

nlp = en_core_web_sm.load()

POS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CONJ",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
    "SPACE",
]


def generate_features(data):
    """Generate features for a list of texts

    Args:
        data (list[str]): the dataset to be processed.

    Returns:
        pandas.DataFrame: the processed features.
    """
    feature_data = []

    for text in data:
        features = preprocess_text(text)
        feature_data.append(features)

    return pd.DataFrame(feature_data)


def preprocess_text(text):
    """Takes a text, generate features, and returns as dict

    Args:
        text (str): the text to be preprocessed.

    Returns:
        dict: a dictionary of feature names with associated values

    """
    text = _simplify_punctuation(text)

    features = {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "smog_index": textstat.smog_index(text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "coleman_liau_index": textstat.coleman_liau_index(text),
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
        "difficult_words": textstat.difficult_words(text),
        "linsear_write_formula": textstat.linsear_write_formula(text),
        "gunning_fog": textstat.gunning_fog(text),
        "text_standard": textstat.text_standard(text, float_output=True),
        "mean_parse_tree_depth": get_mean_parse_tree_depth(text),
        "mean_ents_per_sentence": get_mean_ents_per_sentence(text),
    }

    features.update(get_mean_pos_tags(text))

    return features


def _simplify_punctuation(text):
    # from https://github.com/shivam5992/textstat/issues/77

    text = re.sub(r"[,:;()\-]", " ", text)  # Override commas, colons, etc to spaces/
    text = re.sub(r"[\.!?]", ".", text)  # Change all terminators like ! and ? to "."
    text = re.sub(r"^\s+", "", text)  # Remove white space
    text = re.sub(r"[ ]*(\n|\r\n|\r)[ ]*", " ", text)  # Remove new lines
    text = re.sub(r"([\.])[\. ]+", ".", text)  # Change all ".." to "."
    text = re.sub(r"[ ]*([\.])", ". ", text)  # Normalize all "."`
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces
    text = re.sub(r"\s+$", "", text)  # Remove trailing spaces
    return text


def get_mean_parse_tree_depth(text):
    """Calculate the average depth of parse trees in the text"""
    sentences = text.split(".")
    depths = []
    for doc in list(nlp.pipe(sentences)):
        depths += _get_parse_tree_depths(doc)
    return np.mean(depths)


def _get_parse_tree_depths(doc):
    return [_get_depth(token) for token in doc]


def _get_depth(token, depth=0):
    depths = [_get_depth(child, depth + 1) for child in token.children]
    return max(depths) if len(depths) > 0 else depth


def get_mean_pos_tags(text):
    """Calculate the mean for each type of POS tag in the text"""
    sentences = text.split(".")
    sentence_counts = _make_pos_tag_count_lists(sentences)
    num_sentences = textstat.sentence_count(text)
    mean_pos_tags = _calculate_mean_per_tag(sentence_counts, num_sentences)
    return mean_pos_tags


def _make_pos_tag_count_lists(sentences):
    sentence_counts = {}
    for doc in list(nlp.pipe(sentences)):
        pos_counts = _get_pos_tag_counts(doc)
        for key in pos_counts:
            if key in sentence_counts:
                sentence_counts[key].append(pos_counts[key])
            else:
                sentence_counts[key] = [pos_counts[key]]
    return sentence_counts


def _get_pos_tag_counts(doc):
    pos_counts = {}
    pos_tags = [token.pos_ for token in doc]
    for tag in pos_tags:
        if tag in pos_counts:
            pos_counts[tag] += 1
        else:
            pos_counts[tag] = 1
    return pos_counts


def _calculate_mean_per_tag(counts, num_sentences):
    mean_pos_tags = {f"mean_{tag.lower()}": 0 for tag in POS_TAGS}
    for key in counts:
        if len(counts[key]) < num_sentences:
            counts[key] += [0] * (num_sentences - len(counts[key]))
        mean_value = round(np.mean(counts[key]), 2)
        mean_pos_tags["mean_" + key.lower()] = mean_value
    return mean_pos_tags


def get_total_ents(text):
    """Get the total number of named entities in the text"""
    return len(nlp(text).doc.ents)


def get_mean_ents_per_sentence(text):
    """Calculate the average number of named entities per sentence in the text"""
    return get_total_ents(text) / textstat.sentence_count(text)
