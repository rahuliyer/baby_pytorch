from baby_pytorch.loss import MSE
from baby_pytorch import Value

import pytest

def test_mse_unequal_length():
    predictions = [1, 2, 3]
    targets = [1, 2]

    with pytest.raises(ValueError):
        MSE(predictions, targets)

def test_mse_calculation():
    predictions = [
        Value(1, requires_grad=True),
        Value(2, requires_grad=True),
        Value(3, requires_grad=True)
    ]

    targets = [1, 2, 4]

    expected_loss = ((1 - 1) ** 2 + (2 - 2) ** 2 + (3 - 4) ** 2) / 3

    assert MSE(predictions, targets).data == expected_loss