"""
For the error analysis we need to compare it with the error analysis
from the baseline models but we did not create their error analysis
so we create them in this assignment.

The code for creating an training the baselines models are not modified
except for the import paths and names. The exact same models with the exact same parameters
are produced as in assignment 1.

This directory is probably temporary and used
to get the not yet obtained error analysis metrics for the baselines.
"""

import random
import time

import baselines_error_analysis as error_analysis
import baselines_evaluation as evaluation
import baselines_preprocessing as preprocessing
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

MAX_ITERATIONS = 50000

LOGISTIC_REGRESSION_NAME = "Logistic regression"
SVM_NAME = "SVM"


def main() -> None:
    start_program_time = time.time()

    # Preprocessing

    starting_time = time.time()
    train, dev, test = preprocessing.preprocessing(RANDOM_SEED)
    preprocessing_time = time.time() - starting_time
    print(f"Preprocessing took {preprocessing_time}s")

    starting_time = time.time()
    train_x, train_y, dev_x, dev_y, test_x, test_y = preprocessing.tfidf_generator(
        train, dev, test
    )
    vector_time = time.time() - starting_time
    print(f"Generating vectors took {vector_time}s")

    # Training

    starting_time = time.time()

    logistic_regression = LogisticRegression(
        max_iter=MAX_ITERATIONS, random_state=RANDOM_SEED
    )

    logistic_regression.fit(train_x, train_y)
    dev_prediction_regression = logistic_regression.predict(dev_x)
    test_prediction_regression = logistic_regression.predict(test_x)

    regression_time = time.time() - starting_time
    print(f"{LOGISTIC_REGRESSION_NAME} took {regression_time}s")

    starting_time = time.time()
    svm = LinearSVC()

    svm.fit(train_x, train_y)

    dev_prediction_svm = svm.predict(dev_x)
    test_prediction_svm = svm.predict(test_x)
    svm_time = time.time() - starting_time
    print(f"{SVM_NAME} took {svm_time}s")

    # Evaluation

    evaluation.show_key_metrics(
        test_y, test_prediction_regression, LOGISTIC_REGRESSION_NAME
    )
    evaluation.show_key_metrics(test_y, test_prediction_svm, SVM_NAME)

    # Error analysis

    error_analysis.show_error_analysis(
        test_y, test_prediction_regression, LOGISTIC_REGRESSION_NAME
    )
    error_analysis.show_error_analysis(test_y, test_prediction_svm, SVM_NAME)

    end_program_time = time.time()
    duration_program = end_program_time - start_program_time
    print(f"Total program duration: {duration_program}")


if __name__ == "__main__":
    main()
