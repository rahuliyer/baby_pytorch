from baby_pytorch.tensor import Tensor


def MSE(predictions, targets):
    if not isinstance(predictions, Tensor):
        raise TypeError("predictions must be a Tensor.")
    if not isinstance(targets, Tensor):
        raise TypeError("targets must be a Tensor.")
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")

    return ((predictions - targets) ** 2).mean()
