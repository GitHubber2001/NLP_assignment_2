import os
import re

import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer


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


def tfidf_generator(train_data, dev_data, test_data):
    tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False)
    train_x = tfidf.fit_transform(train_data["text"])
    dev_x = tfidf.transform(dev_data["text"])
    test_x = tfidf.transform(test_data["text"])
    train_y = train_data["label"].values
    dev_y = dev_data["label"].values
    test_y = test_data["label"].values

    return (train_x, train_y, dev_x, dev_y, test_x, test_y)
