import torch

# AG News classes mapping (based on the standard dataset label order)
CLASS_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def print_misclassified_examples(
    model, dataloader, dataset, device, model_name="Model", num_examples=10
):
    """
    Extracts and prints misclassified examples from the test set.
    Assumes the dataloader is NOT shuffled so batch items match dataset indices.
    """
    model.eval()
    errors_found = 0
    current_idx = 0

    print(f"\n{'=' * 60}")
    print(f" ERROR ANALYSIS: {model_name} (First {num_examples} Errors)")
    print(f"{'=' * 60}")

    with torch.no_grad():
        for batch in dataloader:
            x = batch.data.to(device)
            y = batch.labels.to(device)

            # Get predictions
            preds = model(x).argmax(1)

            # Move to CPU for standard python comparison
            preds = preds.cpu().numpy()
            y = y.cpu().numpy()

            for i in range(len(y)):
                if preds[i] != y[i]:
                    # Mismatch found Grab the original text from the dataset
                    original_text = dataset.data.iloc[current_idx]["text"]

                    true_label = CLASS_NAMES.get(y[i], f"Class {y[i]}")
                    pred_label = CLASS_NAMES.get(preds[i], f"Class {preds[i]}")

                    print(f"\n[Mismatch {errors_found + 1}]")
                    print(f"True Label : {true_label}")
                    print(f"Predicted  : {pred_label}")
                    print(f"Text       : {original_text}")

                    errors_found += 1

                    # Stop once we hit the required number of examples
                    if errors_found >= num_examples:
                        print(f"\n{'=' * 60}\n")
                        return

                # Increment index to keep perfectly synced with the dataset
                current_idx += 1
