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
from timer import Timer

# fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BATCH_SIZE = 64
MAX_LENGTH = 100


@Timer.time("Program")
def main() -> None:
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    split_timer = Timer("Split").start()
    train_df, dev_df, test_df = preprocessing.preprocessing(RANDOM_SEED)
    print(train_df[:20])
    split_timer.stop()

    dict_timer = Timer("Dict").start()
    dictionary = preprocessing.generate_vocab(train_df["text"], 2, 50000)
    print(len(dictionary))
    print(list(dictionary.items())[:20])
    dict_timer.stop()

    dataprocessing_timer = Timer("Dataprocessing").start()
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
    dataprocessing_timer.stop()

    lengths = [len(preprocessing.tokenize(text)) for text in train_df["text"]]
    plt.hist(lengths, "auto")
    plt.title("Distribution of tokenized text lengths in training set")
    plt.xlabel("Length of tokenized text")
    plt.ylabel("Frequency")
    plt.show()

    training_time = Timer("training").start()
    cnn_model = models.CNN(len(dictionary), 4).to(device)
    loss_history, accuracy_history = model_training.training(
        cnn_model, train_dataloader, dev_dataloader, device
    )
    plt.plot(loss_history)
    plt.plot(accuracy_history)
    plt.show()
    training_time.stop()


if __name__ == "__main__":
    main()
