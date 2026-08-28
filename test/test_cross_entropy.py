import numpy as np
import pytest
import torch
import torch.nn.functional as torch_functional

from baby_pytorch.loss import cross_entropy
from baby_pytorch.tensor import Tensor


def test_cross_entropy_values_and_gradients_match_pytorch():
    logits_data = np.array(
        [
            [1.2, -0.5, 0.3, 2.1],
            [-1.0, 0.7, 1.5, 0.2],
            [0.1, -0.2, 0.4, 0.8],
        ],
    )
    target_data = np.array([3, 1, 0])
    logits = Tensor(logits_data, requires_grad=True)

    loss = cross_entropy(logits, target_data)
    loss.backward()

    torch_logits = torch.tensor(
        logits_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_loss = torch_functional.cross_entropy(
        torch_logits,
        torch.tensor(target_data),
    )
    torch_loss.backward()

    assert loss.shape == ()
    assert loss.data == pytest.approx(torch_loss.item())
    np.testing.assert_allclose(logits.grad, torch_logits.grad.numpy())


def test_cross_entropy_is_stable_for_extreme_logits():
    logits_data = np.array(
        [
            [1000.0, 999.0, -1000.0],
            [-1000.0, -999.0, 1000.0],
        ],
    )
    targets = [0, 2]
    logits = Tensor(logits_data, requires_grad=True)

    loss = cross_entropy(logits, targets)
    loss.backward()

    torch_logits = torch.tensor(
        logits_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_loss = torch_functional.cross_entropy(
        torch_logits,
        torch.tensor(targets),
    )
    torch_loss.backward()

    assert np.isfinite(loss.data)
    assert np.all(np.isfinite(logits.grad))
    assert loss.data == pytest.approx(torch_loss.item())
    np.testing.assert_allclose(logits.grad, torch_logits.grad.numpy())


def test_cross_entropy_accepts_integer_tensor_targets():
    logits = Tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    targets = Tensor([0, 2], dtype=np.int64)

    tensor_target_loss = cross_entropy(logits, targets)
    list_target_loss = cross_entropy(logits, [0, 2])

    assert tensor_target_loss.data == pytest.approx(list_target_loss.data)


@pytest.mark.parametrize(
    ("logits", "targets", "message"),
    [
        (Tensor([1.0, 2.0]), [0], "logits must have shape"),
        (Tensor([[1.0, 2.0]]), [[0]], "targets must have shape"),
        (Tensor([[1.0, 2.0]]), [0, 1], "targets must have shape"),
        (Tensor([[1.0, 2.0]]), [0.5], "integer class indices"),
        (Tensor([[1.0, 2.0]]), [-1], "outside the valid range"),
        (Tensor([[1.0, 2.0]]), [2], "outside the valid range"),
    ],
)
def test_cross_entropy_rejects_invalid_shapes_and_targets(
    logits,
    targets,
    message,
):
    with pytest.raises(ValueError, match=message):
        cross_entropy(logits, targets)
