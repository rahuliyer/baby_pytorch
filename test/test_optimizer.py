import numpy as np

from baby_pytorch.optim import SGD
from baby_pytorch.tensor import Tensor


def test_zero_grad_resets_each_parameter_to_a_shaped_array():
    matrix = Tensor(np.ones((2, 3)), requires_grad=True)
    vector = Tensor(np.ones(3), requires_grad=True)
    matrix.grad = np.full(matrix.shape, 4.0)
    vector.grad = np.full(vector.shape, -2.0)
    optimizer = SGD([matrix, vector])

    optimizer.zero_grad()

    assert isinstance(matrix.grad, np.ndarray)
    assert isinstance(vector.grad, np.ndarray)
    assert matrix.grad.shape == matrix.shape
    assert vector.grad.shape == vector.shape
    np.testing.assert_array_equal(matrix.grad, np.zeros(matrix.shape))
    np.testing.assert_array_equal(vector.grad, np.zeros(vector.shape))
