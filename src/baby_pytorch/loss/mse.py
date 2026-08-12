def MSE(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")

    num_samples = len(predictions)
    total_loss = sum((pred - target) ** 2
                     for pred, target in zip(predictions, targets))

    return total_loss / num_samples
