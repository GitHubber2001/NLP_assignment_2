"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Nik Skouf (s5617804)
"""

import random
import time

import numpy as np

import error_analysis
import evaluation
import preprocessing
from timer import Timer

# fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def main() -> None:
    program_timer = Timer("Program").Start()

    program_timer.Stop()


if __name__ == "__main__":
    main()
