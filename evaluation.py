import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def display_key_metrics(y_real, y_prediction, model_name: str) -> None:
    """Displays key evaluation metrics"""

    report = classification_report(y_real, y_prediction)

    cm = confusion_matrix(y_real, y_prediction)
    confusion_matrix_display_regression = ConfusionMatrixDisplay(confusion_matrix=cm)
    confusion_matrix_display_regression.plot()
    accuracy_classification_score = accuracy_score(y_real, y_prediction, normalize=True)

    print(f"{model_name}:\n {report}")

    print(
        f"{model_name}: Accuracy classificaion score: {accuracy_classification_score}"
    )

    plt.title(f"Confusion Matrix | {model_name} with TF-IDF")
    plt.show()
