import os
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import torch
from sklearn import model_selection
from torch.utils.data import Dataset

PADDING = "<padding>"
UNKOWN = "<unkown>"


def tokenize(text: str):
    """Returns a tokenized regex pattern"""

    return re.compile(r"[a-z]+").findall(text.lower())


def merge_colums(dataframes: list):
    """Drops and merges the title and description columns into a text column"""

    for dataframe in dataframes:
        dataframe["text"] = dataframe["title"] + " " + dataframe["description"]
        dataframe.drop(columns=["title", "description"], inplace=True)


def preprocessing(random_seed: int):
    """Returns preprocessed train, validation and test sets from the dataset"""

    test_df = pd.read_json(os.path.join("data", "test.jsonl"), lines=True)
    train_df = pd.read_json(os.path.join("data", "train.jsonl"), lines=True)

    train_df, validation_df = model_selection.train_test_split(
        train_df, random_state=random_seed, test_size=0.1, train_size=0.9
    )

    train_df["label"] -= 1
    validation_df["label"] -= 1
    test_df["label"] -= 1

    merge_colums([train_df, validation_df, test_df])

    return (train_df, validation_df, test_df)


def generate_vocab(data, minimum_word_occurance=2, maximum_length=20000):
    """Generates and return a vocabulary from the given data"""

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
    """Returns a list of indices from the tokens and dictionary"""

    return [dictionary.get(token, dictionary[UNKOWN]) for token in tokens]


@dataclass
class Batch:
    data: torch.Tensor
    labels: torch.Tensor
    lengths: torch.Tensor


class TextData(Dataset):
    def __init__(
        self, data: pd.DataFrame, dictionary: dict, max_item_length: int = 100
    ) -> None:
        self.data = data
        self.dictionary = dictionary
        self.max_len = max_item_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datapoint = self.data.iloc[index]
        tokens = tokenize(datapoint["text"])

        return convert_to_indices(tokens, self.dictionary)[: self.max_len], datapoint[
            "label"
        ]

    def collate_fn(self, batch: list[tuple[list[int], int]]) -> Batch:
        id_list = []
        label_list = []
        length_list = []

        for indices, label in batch:
            id_list.append(indices)
            label_list.append(label)
            length_list.append(len(indices))

        data = torch.full(
            (len(batch), self.max_len), self.dictionary[PADDING], dtype=torch.long
        )

        for i, id in enumerate(id_list):
            data[i, : len(id)] = torch.tensor(id)

        label_list = torch.tensor(label_list, dtype=torch.long)
        length_list = torch.tensor(length_list, dtype=torch.long)

        return Batch(data, label_list, length_list)
