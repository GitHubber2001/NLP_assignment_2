"""
Baselines evaluation
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def show_key_metrics(test_y, test_prediction, model_name: str) -> None:
    target_names = ["label", "title", "description"]
    labels = [0, 1, 2, 3]

    report = classification_report(test_y, test_prediction)

    cm = confusion_matrix(test_y, test_prediction)
    confusion_matrix_display_regression = ConfusionMatrixDisplay(confusion_matrix=cm)
    confusion_matrix_display_regression.plot()
    accuracy_classification_score = accuracy_score(
        test_y, test_prediction, normalize=True
    )

    print(f"{model_name}:\n {report}")

    print(
        f"{model_name}: Accuracy classificaion score: {accuracy_classification_score}"
    )

    plt.title(f"Confusion Matrix | {model_name} with TF-IDF")
    plt.show()
