import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader
from models import LSTM

def evaluate(model: nn.Module, data: DataLoader, device: str):
    """Returns an evaluation of the model based on the given data"""

    model.eval()
    size = len(data.dataset)

    true = []
    predictions = []

    with torch.no_grad():
        for batch in data:
            x = batch.data.to(device)
            lengths = batch.lengths.to(device)
            y = batch.labels.to(device)
            prediction = model(x).argmax(1)
            true.append(y.cpu().numpy())
            predictions.append(prediction.cpu().numpy())

    total_true = np.concatenate(true)
    total_prediction = np.concatenate(predictions)

    return accuracy_score(total_true, total_prediction)


def training(
    model: nn.Module,
    training_data: DataLoader,
    validation_data: DataLoader,
    device,
    learning_rate: float = 1e-3,
    max_epochs: int = 30,
    loss_function=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    stopping_accuracy: float = 1e-6,
) -> tuple[list[float], list[float]]:
    """Trains a model and returns the training loss history
    and validation accuracy history"""

    optimizer = optimizer(model.parameters(), lr=learning_rate)
    model.train()
    validation_accuracy = 0
    training_loss_history = []
    validation_accuracy_history = []

    for _ in range(max_epochs):
        last_validation_accuracy = validation_accuracy
        total_loss = 0
        total_length = 0

        for batch in training_data:
            x = batch.data.to(device)
            lengths = batch.lengths.to(device)
            y = batch.labels.to(device)
            prediction = model(x)
            loss = loss_function(prediction, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * y.size(0)
            total_length += y.size(0)

        training_loss = total_loss / total_length
        training_loss_history.append(training_loss)

        if isinstance(model, LSTM):
            # slow but fixes error "RuntimeError: cudnn RNN backward can only be called in training mode"
            model.eval()
            validation_accuracy = evaluate(model, validation_data, device)
            model.train()
        else:
            validation_accuracy = evaluate(model, validation_data, device)

        validation_accuracy_history.append(validation_accuracy)

        if validation_accuracy - last_validation_accuracy < stopping_accuracy:
            break

    return training_loss_history, validation_accuracy_history
