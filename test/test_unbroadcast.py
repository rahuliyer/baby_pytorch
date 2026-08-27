import numpy as np
import pytest

from baby_pytorch.functions import unbroadcast
from baby_pytorch.tensor import Tensor

def test_unbroadcast_leaves_an_already_matching_shape_unchanged():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = unbroadcast(tensor.data, (2, 3))

    np.testing.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])
    assert result.shape == (2, 3)

def test_unbroadcast_removes_a_leading_broadcast_dimension():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = unbroadcast(tensor.data, (3,))

    np.testing.assert_array_equal(result, [5, 7, 9])
    assert result.shape == (3,)

def test_unbroadcast_preserves_a_leading_singleton_dimension():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = unbroadcast(tensor.data, (1, 3))

    np.testing.assert_array_equal(result, [[5, 7, 9]])
    assert result.shape == (1, 3)

def test_unbroadcast_reduces_a_trailing_broadcast_dimension():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = unbroadcast(tensor.data, (2, 1))

    np.testing.assert_array_equal(result, [[6], [15]])
    assert result.shape == (2, 1)

def test_unbroadcast_reduces_multiple_singleton_dimensions():
    tensor = Tensor(np.arange(24).reshape(2, 3, 4))

    result = unbroadcast(tensor.data, (1, 3, 1))

    expected = np.arange(24).reshape(2, 3, 4).sum(axis=(0, 2), keepdims=True)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (1, 3, 1)

def test_unbroadcast_removes_leading_dimensions_and_reduces_singletons():
    tensor = Tensor(np.arange(24).reshape(2, 3, 4))

    result = unbroadcast(tensor.data, (3, 1))

    expected = np.arange(24).reshape(2, 3, 4).sum(axis=0).sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (3, 1)

def test_unbroadcast_reduces_to_a_scalar():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = unbroadcast(tensor.data, ())

    assert result == pytest.approx(21)
    assert result.shape == ()

@pytest.mark.parametrize(
    "target_shape",
    [
        (2, 2),
        (3, 3),
        (1, 2, 3),
    ],
)
def test_unbroadcast_rejects_shapes_that_could_not_have_been_broadcast(target_shape):
    tensor = Tensor(np.ones((2, 3)))

    with pytest.raises(ValueError):
        unbroadcast(tensor.data, target_shape)
