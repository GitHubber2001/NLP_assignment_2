"""
Baselines error analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_error_analysis(test_y, test_prediction, model_name: str) -> None:
    max_amount_errors_to_display = 20

    errors = []
    for i, true_label in enumerate(test_y):
        prediction_label = test_prediction[i]

        if true_label != prediction_label:
            errors.append([true_label, prediction_label])

    amount_errors = len(errors)

    print(f"{model_name}: Total amount errors: {amount_errors}")

    amount_errors_to_display = min(amount_errors, max_amount_errors_to_display)

    for i in range(1, amount_errors_to_display + 1):
        error_pair = errors[i]
        true_label = error_pair[0]
        prediction_label = error_pair[1]

        print(
            f"Error #{i}: True label: {true_label} | Predicted label {prediction_label}"
        )
