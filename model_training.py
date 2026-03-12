import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, data: DataLoader, device) -> float:
    model.eval()
    size = len(data.dataset)

    correct = 0

    with torch.no_grad():
        for batch in data:
            x = batch.data.to(device)
            lengths = batch.lengths.to(device)
            y = batch.labels.to(device)
            prediction = model(x).argmax(1)
            if prediction == y:
                correct += 1
    return correct / size


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
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    model.train()
    validation_accuracy = 0
    training_loss_history = []
    validation_accuracy_history = []

    for epoch in range(max_epochs):
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
        validation_accuracy = evaluate(model, validation_data, device)
        validation_accuracy_history.append(validation_accuracy)
        if validation_accuracy - last_validation_accuracy < stopping_accuracy:
            break
    return training_loss_history, validation_accuracy_history
