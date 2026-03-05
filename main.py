"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Nik Skouf (s5617804)
"""

import random

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


@Timer.Time("Program")
def main() -> None:
    split_timer = Timer("Split").Start()
    train_ds, dev_ds, test_ds = preprocessing.preprocessing(RANDOM_SEED)
    split_timer.Stop()

    dict_timer = Timer("Dict").Start()
    dictionary = preprocessing.generate_vocab(train_ds["text"], 2, 20000)
    print(len(dictionary))
    print(list(dictionary.items())[:20])
    dict_timer.Stop()


if __name__ == "__main__":
    main()
