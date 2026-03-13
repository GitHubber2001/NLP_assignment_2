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
import model_training
import models
import preprocessing
from evaluation import display_key_metrics
from utilities.timer import TimeManager

# fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BATCH_SIZE = 64
MAX_LENGTH = 100


def get_accelerator_device() -> str:
    """Returns accelerator device for boosting performance"""

    current_accelerator = torch.accelerator.current_accelerator(True)
    if current_accelerator is not None:
        device = current_accelerator.type
    else:
        device = "cpu"

    return device


def main() -> None:
    device = get_accelerator_device()
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

    # plot of distribution of lengths tokens
    lengths = [len(preprocessing.tokenize(text)) for text in train_df["text"]]
    plt.figure()
    plt.hist(lengths, "auto")
    plt.title("Distribution of tokenized text lengths in training set")
    plt.xlabel("Length of tokenized text")
    plt.ylabel("Frequency")
    plt.show(block=False)

    vocab_size = len(dictionary)
    amount_classes = 4

    with TimeManager("Training CNN"):
        cnn_model = models.CNN(vocab_size, amount_classes).to(device)
        cnn_loss_history, cnn_accuracy_history = model_training.training(
            cnn_model, train_dataloader, dev_dataloader, device
        )

    with TimeManager("Training LSTM"):
        lstm_model = models.LSTM(vocab_size, amount_classes).to(device)
        lstm_loss_history, lstm_accuracy_history = model_training.training(
            lstm_model, train_dataloader, dev_dataloader, device
        )

    # plot of CNN loss history and accuracy history
    plt.figure()
    plt.title("CNN loss history")
    plt.plot(cnn_loss_history)
    plt.plot(cnn_accuracy_history)
    plt.show(block=False)

    # plot of LSTM loss history and accuracy history
    plt.figure()
    plt.title("LSTM loss history")
    plt.plot(lstm_loss_history)
    plt.plot(lstm_accuracy_history)
    plt.show(block=False)

    with TimeManager("Predicitions"):
        test_data = test_ds.data["text"].tolist()
        test_real_y = test_ds.data["label"].tolist()

        cnn_prediction = cnn_model.forward(test_data)
        lstm_prediction = cnn_model.forward(test_data)

        display_key_metrics(test_real_y, cnn_prediction, "CNN")
        display_key_metrics(test_real_y, lstm_prediction, "LSTM")


if __name__ == "__main__":
    with TimeManager("Program", True):
        main()

    # to keep plots open
    plt.show()
