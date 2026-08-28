import numpy as np

from baby_pytorch.tensor import Tensor


def cross_entropy(logits, targets):
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a Tensor.")
    if len(logits.shape) != 2:
        raise ValueError("logits must have shape (batch_size, num_classes).")

    batch_size, num_classes = logits.shape
    if batch_size == 0 or num_classes == 0:
        raise ValueError("logits must contain at least one sample and class.")

    target_data = targets.data if isinstance(targets, Tensor) else targets
    target_data = np.asarray(target_data)
    if target_data.shape != (batch_size,):
        raise ValueError("targets must have shape (batch_size,).")
    if not np.issubdtype(target_data.dtype, np.number):
        raise ValueError("targets must contain integer class indices.")
    if not np.all(np.isfinite(target_data)) or not np.all(
        target_data == np.floor(target_data)
    ):
        raise ValueError("targets must contain integer class indices.")

    target_indices = target_data.astype(np.int64)
    if np.any(target_indices < 0) or np.any(target_indices >= num_classes):
        raise ValueError("targets contain a class index outside the valid range.")

    max_logits = logits.data.max(axis=-1, keepdims=True)
    logsumexp = (
        (logits - max_logits)
        .exp()
        .sum(dim=-1, keepdims=True)
        .log()
        + max_logits
    )

    target_logits = logits[
        np.arange(batch_size),
        target_indices,
    ].reshape(batch_size, 1)

    return (logsumexp - target_logits).mean()
