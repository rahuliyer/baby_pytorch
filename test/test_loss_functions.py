import numpy as np
import pytest

from baby_pytorch import Tensor
from baby_pytorch.loss import MSE


def test_mse_unequal_length():
    predictions = [1, 2, 3]
    targets = [1, 2]

    with pytest.raises(ValueError):
        MSE(predictions, targets)


def test_mse_calculation():
    predictions = [
        Tensor(1, requires_grad=True),
        Tensor(2, requires_grad=True),
        Tensor(3, requires_grad=True)
    ]

    targets = [1, 2, 4]

    expected_loss = ((1 - 1) ** 2 + (2 - 2) ** 2 + (3 - 4) ** 2) / 3

    assert MSE(predictions, targets).data == expected_loss


def test_mse_accepts_array_backed_prediction_and_target_tensors():
    predictions = Tensor(
        [[1.0, 2.0], [3.0, 5.0]],
        requires_grad=True,
    )
    targets = Tensor([[0.0, 2.5], [4.0, 1.0]])

    loss = MSE(predictions, targets)

    expected = np.mean((predictions.data - targets.data) ** 2)
    assert loss.shape == ()
    assert loss.data == pytest.approx(expected)


def test_batched_mse_gradients_are_averaged_over_every_element():
    predictions = Tensor(
        [[1.0, 2.0], [3.0, 5.0]],
        requires_grad=True,
    )
    targets = Tensor(
        [[0.0, 2.5], [4.0, 1.0]],
        requires_grad=True,
    )

    MSE(predictions, targets).backward()

    expected_prediction_grad = 2 * (predictions.data - targets.data) / 4
    np.testing.assert_allclose(predictions.grad, expected_prediction_grad)
    np.testing.assert_allclose(targets.grad, -expected_prediction_grad)


def test_array_backed_mse_requires_matching_shapes():
    with pytest.raises(
        ValueError,
        match="Predictions and targets must have the same shape",
    ):
        MSE(Tensor([[1.0], [2.0]]), Tensor([1.0, 2.0]))
