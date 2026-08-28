import numpy as np
import pytest

from baby_pytorch import Tensor
from baby_pytorch.loss import MSE


@pytest.mark.parametrize(
    ("predictions", "targets", "message"),
    [
        ([1.0, 2.0], Tensor([1.0, 2.0]), "predictions must be a Tensor"),
        (Tensor([1.0, 2.0]), np.array([1.0, 2.0]), "targets must be a Tensor"),
    ],
)
def test_mse_requires_tensor_inputs(predictions, targets, message):
    with pytest.raises(TypeError, match=message):
        MSE(predictions, targets)


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
