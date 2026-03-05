"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Nik Skouf (s5617804)
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import error_analysis
import evaluation
import preprocessing
from timer import Timer

# fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


@Timer.time("Program")
def main() -> None:

    split_timer = Timer("Split").start()
    train_ds, dev_ds, test_ds = preprocessing.preprocessing(RANDOM_SEED)
    split_timer.stop()

    dict_timer = Timer("Dict").start()
    dictionary = preprocessing.generate_vocab(train_ds["text"], 2, 50000)
    print(len(dictionary))
    print(list(dictionary.items())[:20])
    dict_timer.stop()

    lengths = [len(preprocessing.tokenize(text)) for text in train_ds["text"]]
    plt.hist(lengths, "auto")
    plt.title("Distribution of tokenized text lengths in training set")
    plt.xlabel("Length of tokenized text")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    main()
