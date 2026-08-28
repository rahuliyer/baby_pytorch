from baby_pytorch.tensor import Tensor


def MSE(predictions, targets):
    if isinstance(predictions, Tensor) or isinstance(targets, Tensor):
        predictions = (
            predictions
            if isinstance(predictions, Tensor)
            else Tensor(predictions)
        )
        targets = targets if isinstance(targets, Tensor) else Tensor(targets)

        if predictions.shape != targets.shape:
            raise ValueError(
                "Predictions and targets must have the same shape."
            )

        return ((predictions - targets) ** 2).mean()

    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")

    num_samples = len(predictions)
    total_loss = sum((pred - target) ** 2
                     for pred, target in zip(predictions, targets))

    return total_loss / num_samples
