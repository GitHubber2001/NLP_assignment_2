import torch
from torch import nn
from torch.utils.data import Dataloader


def evaluate(model: nn.Module, data: Dataloader) -> float:
    model.eval()
    size = len(data.dataset)

    correct = 0

    with torch.no_grad():
        for X, y in data:
            prediction = model(X).argmax(1)
            if prediction == y:
                correct += 1
    return correct / size


def training(
    model: nn.Module,
    training_data: Dataloader,
    validation_data: Dataloader,
    learning_rate: float = 1e-3,
    max_epochs: int = 30,
    loss_function=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    stopping_accuracy: float = 1e-6,
) -> tuple[list[float], list[float]]:
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    model.train()
    validation_accuracy = 0
    training_loss_history = []
    validation_accuracy_history = []

    for epoch in range(max_epochs):
        last_validation_accuracy = validation_accuracy
        total_loss = 0
        total_length = 0
        for batch, (X, y) in enumerate(training_data):
            prediction = model(X)
            loss = loss_function(prediction, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * y.size(0)
            total_length += y.size(0)

        training_loss = total_loss / total_length
        training_loss_history.append(training_loss)
        validation_accuracy = evaluate(model, validation_data)
        validation_accuracy_history.append(validation_accuracy)
        if validation_accuracy - last_validation_accuracy < stopping_accuracy:
            break
    return training_loss_history, validation_accuracy_history
