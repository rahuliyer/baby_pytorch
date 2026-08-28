import numpy as np
import pytest

from baby_pytorch.tensor import Tensor


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (lambda left, right: left + right, [[11, 22, 33], [14, 25, 36]]),
        (lambda left, right: left - right, [[-9, -18, -27], [-6, -15, -24]]),
        (lambda left, right: left * right, [[10, 40, 90], [40, 100, 180]]),
        (lambda left, right: left / right, [[0.1, 0.1, 0.1], [0.4, 0.25, 0.2]]),
    ],
)
def test_binary_operations_broadcast_a_row_across_a_matrix(operation, expected):
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    row = Tensor([10, 20, 30])

    result = operation(matrix, row)

    np.testing.assert_allclose(result.data, expected)
    assert result.shape == (2, 3)


def test_power_broadcasts_an_exponent_row_across_a_matrix():
    base = Tensor([[1, 2, 3], [4, 5, 6]])
    exponent = Tensor([1, 2, 3])

    result = base**exponent

    np.testing.assert_array_equal(result.data, [[1, 4, 27], [4, 25, 216]])


def test_broadcasts_a_column_and_a_row_to_a_matrix():
    column = Tensor([[1], [2], [3]])
    row = Tensor([10, 20, 30, 40])

    result = column + row

    np.testing.assert_array_equal(
        result.data,
        [[11, 21, 31, 41], [12, 22, 32, 42], [13, 23, 33, 43]],
    )
    assert result.shape == (3, 4)


def test_broadcasts_a_scalar_tensor_and_python_scalar():
    values = Tensor([[1, 2], [3, 4]])

    np.testing.assert_array_equal((values + Tensor(2)).data, [[3, 4], [5, 6]])
    np.testing.assert_array_equal((2 * values).data, [[2, 4], [6, 8]])
    np.testing.assert_allclose((12 / values).data, [[12, 6], [4, 3]])


def test_broadcasts_across_multiple_leading_dimensions():
    values = Tensor(np.arange(24).reshape(2, 3, 4))
    offsets = Tensor([100, 200, 300, 400])

    result = values + offsets

    np.testing.assert_array_equal(result.data, values.data + offsets.data)
    assert result.shape == (2, 3, 4)


def test_incompatible_shapes_raise_a_value_error():
    with pytest.raises(ValueError):
        Tensor(np.zeros((2, 3))) + Tensor(np.zeros((2, 2)))


@pytest.mark.parametrize(
    "other_data",
    [
        pytest.param(np.array([1, 2, 1, 2]), id="row"),
        pytest.param(np.array([[1, 2, 1, 2]]), id="singleton-row"),
        pytest.param(np.array([[1], [2], [3]]), id="column"),
        pytest.param(np.array(2.0), id="scalar"),
    ],
)
@pytest.mark.parametrize(
    ("tensor_operation", "numpy_operation"),
    [
        pytest.param(lambda left, right: left + right, np.add, id="add"),
        pytest.param(lambda left, right: left - right, np.subtract, id="subtract"),
        pytest.param(lambda left, right: left * right, np.multiply, id="multiply"),
        pytest.param(lambda left, right: left / right, np.divide, id="divide"),
        pytest.param(lambda left, right: left**right, np.power, id="power"),
    ],
)
def test_elementwise_operations_broadcast_either_operand(
    other_data,
    tensor_operation,
    numpy_operation,
):
    matrix_data = np.arange(1, 13).reshape(3, 4)
    matrix = Tensor(matrix_data)
    other = Tensor(other_data)

    forward = tensor_operation(matrix, other)
    reverse = tensor_operation(other, matrix)

    np.testing.assert_allclose(
        forward.data,
        numpy_operation(matrix_data, other_data),
    )
    np.testing.assert_allclose(
        reverse.data,
        numpy_operation(other_data, matrix_data),
    )
    assert forward.shape == (3, 4)
    assert reverse.shape == (3, 4)
