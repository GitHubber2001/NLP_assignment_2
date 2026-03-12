"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Nik Skouf (s5617804)
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import error_analysis
import evaluation
import model_training
import models
import preprocessing
from utilities.timer import TimeManager

# fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BATCH_SIZE = 64
MAX_LENGTH = 100


def main() -> None:
    current_accelerator = torch.accelerator.current_accelerator(True)
    if current_accelerator is not None:
        device = current_accelerator.type
    else:
        device = "cpu"

    print(f"Using {device} device")

    with TimeManager("Split"):
        train_df, dev_df, test_df = preprocessing.preprocessing(RANDOM_SEED)

        print(train_df[:20])

    with TimeManager("Dictionary"):
        dictionary = preprocessing.generate_vocab(train_df["text"], 2, 50000)

        print(len(dictionary))
        print(list(dictionary.items())[:20])

    with TimeManager("Data processing"):
        train_ds = preprocessing.TextData(train_df, dictionary, MAX_LENGTH)
        dev_ds = preprocessing.TextData(dev_df, dictionary, MAX_LENGTH)
        test_ds = preprocessing.TextData(test_df, dictionary, MAX_LENGTH)

        train_dataloader = DataLoader(
            train_ds, BATCH_SIZE, True, collate_fn=train_ds.collate_fn
        )
        dev_dataloader = DataLoader(
            dev_ds, BATCH_SIZE, True, collate_fn=dev_ds.collate_fn
        )
        test_dataloader = DataLoader(
            test_ds, BATCH_SIZE, True, collate_fn=test_ds.collate_fn
        )

    # plot of istribution of lengths tokens
    lengths = [len(preprocessing.tokenize(text)) for text in train_df["text"]]
    plt.figure()
    plt.hist(lengths, "auto")
    plt.title("Distribution of tokenized text lengths in training set")
    plt.xlabel("Length of tokenized text")
    plt.ylabel("Frequency")
    plt.show(block=False)

    with TimeManager("Training"):
        cnn_model = models.CNN(len(dictionary), 4).to(device)

        loss_history, accuracy_history = model_training.training(
            cnn_model, train_dataloader, dev_dataloader, device
        )

    # plot of loss history and accuracy history
    plt.figure()
    plt.plot(loss_history)
    plt.plot(accuracy_history)
    plt.show(block=False)


if __name__ == "__main__":
    with TimeManager("Program"):
        main()

    # to keep plots open
    plt.show()
