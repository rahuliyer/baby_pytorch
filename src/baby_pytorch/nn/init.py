import math

import numpy as np


def calculate_gain(nonlinearity):
    """Return the recommended initialization gain for a nonlinearity."""
    gains = {
        "relu": math.sqrt(2.0),
        "sigmoid": 1.0,
        "tanh": 5.0 / 3.0,
    }
    if not isinstance(nonlinearity, str) or nonlinearity not in gains:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity!r}")
    return gains[nonlinearity]


def kaiming_normal_(tensor, fan_in, nonlinearity):
    """Fill a tensor in place using Kaiming normal initialization."""
    if (
        not isinstance(fan_in, (int, np.integer))
        or isinstance(fan_in, (bool, np.bool_))
        or fan_in <= 0
    ):
        raise ValueError("fan_in must be a positive integer.")

    std = calculate_gain(nonlinearity) / math.sqrt(fan_in)
    tensor.data[...] = np.random.normal(0.0, std, size=tensor.shape)
    return tensor
