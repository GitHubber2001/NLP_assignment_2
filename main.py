"""
Kevin Kuipers (s5051150)
Federico Berdugo Morales (s5363268)
Nikolaos Skoufis (s5617804)
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import error_analysis
import model_training
import models
import preprocessing
from evaluation import display_key_metrics
from utilities.timer import TimeManager

# fixed random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 64
MAX_LENGTH = 100


def get_accelerator_device() -> str:
    """Returns accelerator device for boosting performance"""

    current_accelerator = torch.accelerator.current_accelerator(True)
    if current_accelerator is not None:
        device = current_accelerator.type
    else:
        device = "cpu"

    return device


def main() -> None:
    device = get_accelerator_device()
    print(f"Using {device} device")

    with TimeManager("Split"):
        train_df, dev_df, test_df = preprocessing.preprocessing(RANDOM_SEED)

    with TimeManager("Dictionary"):
        dictionary = preprocessing.generate_vocab(train_df["text"], 2, 50000)

    with TimeManager("Data processing"):
        train_ds = preprocessing.TextData(train_df, dictionary, MAX_LENGTH)
        dev_ds = preprocessing.TextData(dev_df, dictionary, MAX_LENGTH)
        test_ds = preprocessing.TextData(test_df, dictionary, MAX_LENGTH)

        train_dataloader = DataLoader(
            train_ds, BATCH_SIZE, True, collate_fn=train_ds.collate_fn
        )
        dev_dataloader = DataLoader(
            dev_ds,
            BATCH_SIZE,
            collate_fn=dev_ds.collate_fn,  # Removed shuffle=True for eval sets
        )
        test_dataloader = DataLoader(
            test_ds,
            BATCH_SIZE,
            collate_fn=test_ds.collate_fn,  # Removed shuffle=True for eval sets
        )

    # plot of distribution of lengths tokens
    lengths = [len(preprocessing.tokenize(text)) for text in train_df["text"]]
    plt.figure()
    plt.hist(lengths, "auto")
    plt.title("Distribution of tokenized text lengths in training set")
    plt.xlabel("Length of tokenized text")
    plt.ylabel("Frequency")
    plt.show(block=False)

    vocab_size = len(dictionary)
    amount_classes = 4

    # 1. Train Baseline CNN
    with TimeManager("Training CNN Baseline"):
        cnn_model = models.CNN(vocab_size, amount_classes).to(device)
        cnn_loss_history, cnn_accuracy_history = model_training.training(
            cnn_model, train_dataloader, dev_dataloader, device
        )

    # 2. Train Baseline LSTM (Default Dropout = 0.5)
    with TimeManager("Training LSTM Baseline"):
        lstm_model = models.LSTM(vocab_size, amount_classes).to(device)
        lstm_loss_history, lstm_accuracy_history = model_training.training(
            lstm_model, train_dataloader, dev_dataloader, device
        )

    # 3. Train Ablation LSTM (Dropout = 0.0)
    with TimeManager("Training LSTM Ablation (No Dropout)"):
        # We pass dropout=0.0 to test the effect of removing regularization
        lstm_ablation_model = models.LSTM(vocab_size, amount_classes, dropout=0.0).to(
            device
        )
        ablation_loss_history, ablation_accuracy_history = model_training.training(
            lstm_ablation_model, train_dataloader, dev_dataloader, device
        )

    # Plot CNN Learning Curves
    plt.figure()
    plt.title("CNN Learning Curves")
    plt.plot(cnn_loss_history, label="Train Loss")
    plt.plot(cnn_accuracy_history, label="Dev Accuracy")
    plt.legend()
    plt.show(block=False)

    # Plot LSTM Baseline vs Ablation Learning Curves (Great for the report!)
    plt.figure()
    plt.title("LSTM Regularization Ablation: Dropout 0.5 vs 0.0")
    plt.plot(
        lstm_loss_history,
        label="Baseline Train Loss (Drop 0.5)",
        linestyle="--",
        color="blue",
    )
    plt.plot(
        ablation_loss_history,
        label="Ablation Train Loss (Drop 0.0)",
        linestyle="-",
        color="blue",
    )
    plt.plot(
        lstm_accuracy_history,
        label="Baseline Dev Acc (Drop 0.5)",
        linestyle="--",
        color="orange",
    )
    plt.plot(
        ablation_accuracy_history,
        label="Ablation Dev Acc (Drop 0.0)",
        linestyle="-",
        color="orange",
    )
    plt.legend()
    plt.show(block=False)

    # Final Evaluation & Predictions
    with TimeManager("Predictions & Evaluation"):
        # Helper function to get batched predictions
        def get_test_predictions(model, dataloader):
            model.eval()
            all_preds = []
            all_true = []
            with torch.no_grad():
                for batch in dataloader:
                    x = batch.data.to(device)
                    y = batch.labels.to(device)
                    preds = model(x).argmax(1)
                    all_preds.extend(preds.cpu().numpy())
                    all_true.extend(y.cpu().numpy())
            return all_true, all_preds

        print("\n--- Evaluating CNN Baseline ---")
        test_real_y, cnn_prediction = get_test_predictions(cnn_model, test_dataloader)
        display_key_metrics(test_real_y, cnn_prediction, "CNN Baseline")

        print("\n--- Evaluating LSTM Baseline ---")
        _, lstm_prediction = get_test_predictions(lstm_model, test_dataloader)
        display_key_metrics(test_real_y, lstm_prediction, "LSTM Baseline")

        print("\n--- Evaluating LSTM Ablation (No Dropout) ---")
        _, ablation_prediction = get_test_predictions(
            lstm_ablation_model, test_dataloader
        )
        display_key_metrics(test_real_y, ablation_prediction, "LSTM Ablation")

        with TimeManager("Error Analysis Extraction"):
            # Extract 10 errors from the best performing model (e.g., Baseline LSTM)
            error_analysis.print_misclassified_examples(
                model=lstm_model,
                dataloader=test_dataloader,
                dataset=test_ds,
                device=device,
                model_name="LSTM Baseline",
                num_examples=10,
            )


if __name__ == "__main__":
    with TimeManager("Program", True):
        main()

    # to keep plots open
    plt.show()
