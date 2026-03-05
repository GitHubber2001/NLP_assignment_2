import os
import re
from collections import Counter

import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer

PADDING = "<padding>"
UNKOWN = "<unkown>"


def tokenize(text: str):
    return re.compile(r"[a-z]+").findall(text.lower())


def merge_colums(dataframes: list):
    for dataframe in dataframes:
        dataframe["text"] = dataframe["title"] + " " + dataframe["description"]
        dataframe.drop(columns=["title", "description"], inplace=True)


def preprocessing(random_seed: int):
    test_df = pd.read_json(os.path.join("data", "test.jsonl"), lines=True)
    train_df = pd.read_json(os.path.join("data", "train.jsonl"), lines=True)

    train_df, validation_df = model_selection.train_test_split(
        train_df, random_state=random_seed, test_size=0.1, train_size=0.9
    )

    merge_colums([train_df, validation_df, test_df])
    return (train_df, validation_df, test_df)


def generate_vocab(data, minimum_word_occurance=2, maximum_length=20000):
    counter = Counter()
    for sentence in data:
        counter.update(tokenize(sentence))

    vocab = {PADDING: 0, UNKOWN: 1}

    for word, frequency in counter.most_common():
        if frequency < minimum_word_occurance:
            break
        elif len(vocab) >= maximum_length:
            break
        else:
            vocab[word] = len(vocab)
    return vocab


def convert_to_indices(tokens: list, dictionary: dict) -> list:
    return [dictionary.get(token, dictionary[UNKOWN]) for token in tokens]
