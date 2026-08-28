import math

import numpy as np
import pytest

from baby_pytorch.tensor import Tensor
from baby_pytorch.activation_functions import tanh, sigmoid, relu


def test_chain_rule_for_all_binary_operations():
    x = Tensor(2.0, requires_grad=True)
    y = 3 * ((x + 1) * (x - 4))

    y.backward()

    assert x.grad == pytest.approx(3.0)


def test_power_gradients_for_base_and_exponent():
    base = Tensor(2.0, requires_grad=True)
    exponent = Tensor(3.0, requires_grad=True)

    (4 * (base**exponent)).backward()

    assert base.grad == pytest.approx(48.0)
    assert exponent.grad == pytest.approx(32.0 * math.log(2.0))


def test_shared_nodes_are_processed_once():
    x = Tensor(3.0, requires_grad=True)
    square = x * x

    (square + square).backward()

    assert x.grad == pytest.approx(12.0)


def test_scalar_division_and_reverse_arithmetic():
    x = Tensor(2.0, requires_grad=True)

    y = (10 - x) + (12 / x) + (x / 2)
    y.backward()

    assert y.data == pytest.approx(15.0)
    assert x.grad == pytest.approx(-3.5)


def test_tensors_do_not_require_grad_by_default():
    assert not Tensor(2.0).requires_grad


def test_result_does_not_require_grad_when_operands_do_not():
    left = Tensor(2.0, requires_grad=False)
    right = Tensor(3.0, requires_grad=False)

    assert not (left + right).requires_grad
    assert not (left * right).requires_grad
    assert not (left**right).requires_grad


def test_result_requires_grad_when_any_operand_does():
    tracked = Tensor(2.0, requires_grad=True)
    plain = Tensor(3.0)

    assert (plain + tracked).requires_grad
    assert (tracked * plain).requires_grad
    assert (plain**tracked).requires_grad


def test_repeated_backward_accumulates_leaf_gradients_linearly():
    x = Tensor(2.0, requires_grad=True)
    y = (x * 2) * 3

    y.backward()
    y.backward()

    assert x.grad == pytest.approx(12.0)

def test_activation_function_backward():
    for v, f, expected in [(2.0, tanh, 0.7065082485316443),
                    (2.0, sigmoid, 1.0499358540350662),
                    (2.0, relu, 10.0),
                    (-2.0, relu, 0)]:
        x = Tensor(v, requires_grad=True)
        (10 * f(x)).backward()
        assert x.grad == pytest.approx(expected)


def test_addition_backward_reduces_a_broadcast_row_gradient():
    matrix = Tensor(np.ones((2, 3)), requires_grad=True)
    row = Tensor([10, 20, 30], requires_grad=True)

    (matrix + row).backward()

    np.testing.assert_array_equal(matrix.grad, np.ones((2, 3)))
    np.testing.assert_array_equal(row.grad, [2, 2, 2])
    assert matrix.grad.shape == matrix.shape
    assert row.grad.shape == row.shape


def test_subtraction_backward_reduces_and_negates_a_broadcast_gradient():
    matrix = Tensor(np.ones((2, 3)), requires_grad=True)
    row = Tensor([10, 20, 30], requires_grad=True)

    (matrix - row).backward()

    np.testing.assert_array_equal(matrix.grad, np.ones((2, 3)))
    np.testing.assert_array_equal(row.grad, [-2, -2, -2])


def test_multiplication_backward_reduces_row_and_column_gradients():
    column = Tensor([[1], [2]], requires_grad=True)
    row = Tensor([10, 20, 30], requires_grad=True)

    (column * row).backward()

    np.testing.assert_array_equal(column.grad, [[60], [60]])
    np.testing.assert_array_equal(row.grad, [3, 3, 3])
    assert column.grad.shape == column.shape
    assert row.grad.shape == row.shape


def test_multiplication_backward_reduces_to_a_scalar_gradient():
    values = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    scale = Tensor(2, requires_grad=True)

    (values * scale).backward()

    np.testing.assert_array_equal(values.grad, np.full((2, 3), 2.0))
    np.testing.assert_array_equal(scale.grad, np.array(21.0))
    assert scale.grad.shape == ()


def test_addition_backward_reduces_multiple_leading_dimensions():
    values = Tensor(np.ones((2, 3, 4)), requires_grad=True)
    offsets = Tensor([10, 20, 30, 40], requires_grad=True)

    (values + offsets).backward()

    np.testing.assert_array_equal(values.grad, np.ones((2, 3, 4)))
    np.testing.assert_array_equal(offsets.grad, [6, 6, 6, 6])


def test_division_backward_reduces_a_broadcast_denominator_gradient():
    numerator = Tensor([[2, 4, 6], [8, 10, 12]], requires_grad=True)
    denominator = Tensor([2, 2, 3], requires_grad=True)

    (numerator / denominator).backward()

    np.testing.assert_allclose(
        numerator.grad,
        [[0.5, 0.5, 1 / 3], [0.5, 0.5, 1 / 3]],
    )
    np.testing.assert_allclose(denominator.grad, [-2.5, -3.5, -2.0])


def test_power_backward_reduces_broadcast_base_and_exponent_gradients():
    base = Tensor([[2], [3]], requires_grad=True)
    exponent = Tensor([1, 2, 3], requires_grad=True)

    (base**exponent).backward()

    np.testing.assert_allclose(base.grad, [[17], [34]])
    np.testing.assert_allclose(
        exponent.grad,
        [
            2 * np.log(2) + 3 * np.log(3),
            4 * np.log(2) + 9 * np.log(3),
            8 * np.log(2) + 27 * np.log(3),
        ],
    )


def test_backward_leaves_non_broadcast_gradient_shapes_unchanged():
    left = Tensor([[1, 2], [3, 4]], requires_grad=True)
    right = Tensor([[5, 6], [7, 8]], requires_grad=True)

    (left * right).backward()

    np.testing.assert_array_equal(left.grad, right.data)
    np.testing.assert_array_equal(right.grad, left.data)
    assert left.grad.shape == left.shape
    assert right.grad.shape == right.shape


def test_backward_handles_multiple_broadcasts_in_one_graph():
    matrix = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    row = Tensor([10, 20, 30], requires_grad=True)
    column = Tensor([[100], [200]], requires_grad=True)

    result = matrix * row + column
    result.backward()

    np.testing.assert_array_equal(matrix.grad, [[10, 20, 30], [10, 20, 30]])
    np.testing.assert_array_equal(row.grad, [5, 7, 9])
    np.testing.assert_array_equal(column.grad, [[3], [3]])


def test_repeated_backward_accumulates_reduced_leaf_gradients():
    matrix = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    row = Tensor([10, 20, 30], requires_grad=True)
    result = matrix + row

    result.backward()
    result.backward()

    np.testing.assert_array_equal(matrix.grad, np.full((2, 3), 2.0))
    np.testing.assert_array_equal(row.grad, [4, 4, 4])


def test_log_backward():
    x = Tensor(4.0, requires_grad=True)

    (3 * x.log()).backward()

    assert x.grad == pytest.approx(3.0 / 4.0)


def test_log10_backward():
    x = Tensor(4.0, requires_grad=True)

    (3 * x.log10()).backward()

    assert x.grad == pytest.approx(3.0 / (4.0 * math.log(10)))


def test_exp_backward():
    x = Tensor(2.0, requires_grad=True)

    (3 * x.exp()).backward()

    assert x.grad == pytest.approx(3.0 * math.exp(2.0))


def test_log_backward_elementwise_over_an_array():
    x = Tensor([1.0, 2.0, 4.0], requires_grad=True)

    x.log().backward()

    np.testing.assert_allclose(x.grad, [1.0, 0.5, 0.25])
    assert x.grad.shape == x.shape


def test_exp_backward_uses_upstream_gradient_from_a_larger_graph():
    x = Tensor([0.0, 1.0], requires_grad=True)
    weights = Tensor([2.0, 5.0])

    (x.exp() * weights).backward()

    np.testing.assert_allclose(x.grad, [2.0 * math.exp(0.0), 5.0 * math.exp(1.0)])


def test_composed_log_of_exp_is_identity_gradient():
    x = Tensor(3.0, requires_grad=True)

    x.exp().log().backward()

    assert x.grad == pytest.approx(1.0)


def test_matmul_backward_for_vector_dot_product():
    left = Tensor([1, 2, 3], requires_grad=True)
    right = Tensor([4, 5, 6], requires_grad=True)

    (left @ right).backward()

    np.testing.assert_array_equal(left.grad, right.data)
    np.testing.assert_array_equal(right.grad, left.data)


def test_matmul_backward_for_matrix_by_vector():
    matrix = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    vector = Tensor([7, 8, 9], requires_grad=True)

    (matrix @ vector).backward()

    np.testing.assert_array_equal(matrix.grad, [[7, 8, 9], [7, 8, 9]])
    np.testing.assert_array_equal(vector.grad, [5, 7, 9])
    assert matrix.grad.shape == matrix.shape
    assert vector.grad.shape == vector.shape


def test_matmul_backward_for_vector_by_matrix():
    vector = Tensor([1, 2], requires_grad=True)
    matrix = Tensor([[3, 4, 5], [6, 7, 8]], requires_grad=True)

    (vector @ matrix).backward()

    np.testing.assert_array_equal(vector.grad, [12, 21])
    np.testing.assert_array_equal(matrix.grad, [[1, 1, 1], [2, 2, 2]])


def test_matmul_backward_for_matrix_by_matrix():
    left = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    right = Tensor([[7, 8], [9, 10], [11, 12]], requires_grad=True)

    (left @ right).backward()

    np.testing.assert_array_equal(left.grad, [[15, 19, 23], [15, 19, 23]])
    np.testing.assert_array_equal(right.grad, [[5, 5], [7, 7], [9, 9]])
    assert left.grad.shape == left.shape
    assert right.grad.shape == right.shape


def test_matmul_backward_for_batched_matrices():
    left_data = np.arange(12).reshape(2, 2, 3)
    right_data = np.arange(12).reshape(2, 3, 2)
    left = Tensor(left_data, requires_grad=True)
    right = Tensor(right_data, requires_grad=True)

    result = left @ right
    result.backward()

    upstream = np.ones(result.shape)
    expected_left = np.matmul(upstream, np.swapaxes(right_data, -1, -2))
    expected_right = np.matmul(np.swapaxes(left_data, -1, -2), upstream)
    np.testing.assert_array_equal(left.grad, expected_left)
    np.testing.assert_array_equal(right.grad, expected_right)


def test_matmul_backward_for_batched_matrices_by_a_shared_vector():
    matrices = Tensor(
        np.arange(12).reshape(2, 2, 3),
        requires_grad=True,
    )
    vector = Tensor([1, 2, 3], requires_grad=True)

    result = matrices @ vector
    result.backward()

    np.testing.assert_array_equal(
        matrices.grad,
        [
            [[1, 2, 3], [1, 2, 3]],
            [[1, 2, 3], [1, 2, 3]],
        ],
    )
    np.testing.assert_array_equal(vector.grad, [18, 22, 26])
    assert matrices.grad.shape == matrices.shape
    assert vector.grad.shape == vector.shape


def test_matmul_backward_for_a_shared_vector_by_batched_matrices():
    vector = Tensor([1, 2, 3], requires_grad=True)
    matrices = Tensor(
        np.arange(12).reshape(2, 3, 2),
        requires_grad=True,
    )

    result = vector @ matrices
    result.backward()

    np.testing.assert_array_equal(vector.grad, [14, 22, 30])
    np.testing.assert_array_equal(
        matrices.grad,
        [
            [[1, 1], [2, 2], [3, 3]],
            [[1, 1], [2, 2], [3, 3]],
        ],
    )
    assert vector.grad.shape == vector.shape
    assert matrices.grad.shape == matrices.shape


def test_matmul_backward_reduces_a_broadcast_batch_gradient():
    left_data = np.arange(12).reshape(2, 2, 3)
    right_data = np.arange(6).reshape(3, 2)
    left = Tensor(left_data, requires_grad=True)
    right = Tensor(right_data, requires_grad=True)

    result = left @ right
    result.backward()

    upstream = np.ones(result.shape)
    expected_left = np.matmul(upstream, right_data.T)
    expected_right = np.matmul(
        np.swapaxes(left_data, -1, -2), upstream
    ).sum(axis=0)
    np.testing.assert_array_equal(left.grad, expected_left)
    np.testing.assert_array_equal(right.grad, expected_right)
    assert right.grad.shape == right.shape


def test_matmul_backward_uses_upstream_gradient_from_a_larger_graph():
    left = Tensor([[1, 2], [3, 4]], requires_grad=True)
    right = Tensor([[5, 6], [7, 8]], requires_grad=True)
    weights = Tensor([[1, 2], [3, 4]])

    result = (left @ right) * weights
    result.backward()

    np.testing.assert_array_equal(left.grad, weights.data @ right.data.T)
    np.testing.assert_array_equal(right.grad, left.data.T @ weights.data)


def test_vector_matmul_backward_uses_nonuniform_upstream_gradient():
    vector = Tensor([1, 2], requires_grad=True)
    matrix = Tensor([[3, 4, 5], [6, 7, 8]], requires_grad=True)
    weights = Tensor([2, 3, 4])

    result = (vector @ matrix) * weights
    result.backward()

    np.testing.assert_array_equal(vector.grad, [38, 65])
    np.testing.assert_array_equal(matrix.grad, [[2, 3, 4], [4, 6, 8]])


def test_matmul_backward_reduces_broadcast_dimensions_for_both_operands():
    left_data = np.arange(12).reshape(2, 1, 2, 3)
    right_data = np.arange(24).reshape(1, 4, 3, 2)
    left = Tensor(left_data, requires_grad=True)
    right = Tensor(right_data, requires_grad=True)

    result = left @ right
    result.backward()

    upstream = np.ones(result.shape)
    expected_left = np.matmul(
        upstream,
        np.swapaxes(right_data, -1, -2),
    ).sum(axis=1, keepdims=True)
    expected_right = np.matmul(
        np.swapaxes(left_data, -1, -2),
        upstream,
    ).sum(axis=0, keepdims=True)
    np.testing.assert_array_equal(left.grad, expected_left)
    np.testing.assert_array_equal(right.grad, expected_right)
    assert left.grad.shape == left.shape
    assert right.grad.shape == right.shape


def test_matmul_backward_only_updates_operands_that_require_grad():
    left = Tensor([[1, 2], [3, 4]], requires_grad=True)
    right = Tensor([[5, 6], [7, 8]], requires_grad=False)

    (left @ right).backward()

    np.testing.assert_array_equal(left.grad, [[11, 15], [11, 15]])
    np.testing.assert_array_equal(right.grad, np.zeros((2, 2)))


def test_repeated_matmul_backward_accumulates_leaf_gradients():
    left = Tensor([[1, 2], [3, 4]], requires_grad=True)
    right = Tensor([[5, 6], [7, 8]], requires_grad=True)
    result = left @ right

    result.backward()
    result.backward()

    np.testing.assert_array_equal(left.grad, [[22, 30], [22, 30]])
    np.testing.assert_array_equal(right.grad, [[8, 8], [12, 12]])


def test_reshape_backward_restores_the_original_gradient_shape():
    tensor = Tensor(np.arange(6), requires_grad=True)

    tensor.reshape(2, 3).backward()

    np.testing.assert_array_equal(tensor.grad, np.ones(6))
    assert tensor.grad.shape == tensor.shape


def test_reshape_backward_uses_upstream_gradient_from_a_larger_graph():
    tensor = Tensor(np.arange(6), requires_grad=True)
    weights = Tensor([[1, 2, 3], [4, 5, 6]])

    (tensor.reshape(2, 3) * weights).backward()

    np.testing.assert_array_equal(tensor.grad, [1, 2, 3, 4, 5, 6])


def test_reshape_backward_supports_an_inferred_dimension():
    tensor = Tensor(np.arange(12), requires_grad=True)
    weights = Tensor(np.arange(1, 13).reshape(3, 4))

    (tensor.reshape(3, -1) * weights).backward()

    np.testing.assert_array_equal(tensor.grad, np.arange(1, 13))


def test_backward_through_multiple_reshapes():
    tensor = Tensor(np.arange(6), requires_grad=True)
    weights = Tensor([[1, 2], [3, 4], [5, 6]])

    reshaped = tensor.reshape(2, 3).reshape(3, 2)
    (reshaped * weights).backward()

    np.testing.assert_array_equal(tensor.grad, [1, 2, 3, 4, 5, 6])


def test_repeated_reshape_backward_accumulates_leaf_gradients():
    tensor = Tensor(np.arange(6), requires_grad=True)
    result = tensor.reshape(2, 3)

    result.backward()
    result.backward()

    np.testing.assert_array_equal(tensor.grad, np.full(6, 2.0))


def test_view_backward_restores_the_original_gradient_shape():
    tensor = Tensor(np.arange(6), requires_grad=True)

    tensor.view(2, 3).backward()

    np.testing.assert_array_equal(tensor.grad, np.ones(6))
    assert tensor.grad.shape == tensor.shape


def test_view_backward_uses_upstream_gradient_from_a_larger_graph():
    tensor = Tensor(np.arange(6), requires_grad=True)
    weights = Tensor([[1, 2, 3], [4, 5, 6]])

    (tensor.view(2, 3) * weights).backward()

    np.testing.assert_array_equal(tensor.grad, [1, 2, 3, 4, 5, 6])


def test_backward_through_view_and_reshape():
    tensor = Tensor(np.arange(6), requires_grad=True)
    weights = Tensor([[1, 2], [3, 4], [5, 6]])

    result = tensor.view(2, 3).reshape(3, 2)
    (result * weights).backward()

    np.testing.assert_array_equal(tensor.grad, [1, 2, 3, 4, 5, 6])


def test_repeated_view_backward_accumulates_leaf_gradients():
    tensor = Tensor(np.arange(6), requires_grad=True)
    result = tensor.view(2, 3)

    result.backward()
    result.backward()

    np.testing.assert_array_equal(tensor.grad, np.full(6, 2.0))


def test_swapaxes_backward_restores_the_original_gradient_axes():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
    weights = Tensor([[1, 2], [3, 4], [5, 6]])

    (tensor.swapaxes(0, 1) * weights).backward()

    np.testing.assert_array_equal(tensor.grad, [[1, 3, 5], [2, 4, 6]])
    assert tensor.grad.shape == tensor.shape


def test_swapaxes_backward_supports_negative_axes_on_a_3d_tensor():
    data = np.arange(24).reshape(2, 3, 4)
    weights = np.arange(1, 25).reshape(2, 4, 3)
    tensor = Tensor(data, requires_grad=True)

    (tensor.swapaxes(-1, -2) * Tensor(weights)).backward()

    np.testing.assert_array_equal(tensor.grad, np.swapaxes(weights, -1, -2))
    assert tensor.grad.shape == tensor.shape


def test_backward_through_two_swapaxes_operations():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
    weights = Tensor([[1, 2, 3], [4, 5, 6]])

    result = tensor.swapaxes(0, 1).swapaxes(0, 1)
    (result * weights).backward()

    np.testing.assert_array_equal(tensor.grad, weights.data)


def test_repeated_swapaxes_backward_accumulates_leaf_gradients():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
    result = tensor.swapaxes(0, 1)

    result.backward()
    result.backward()

    np.testing.assert_array_equal(tensor.grad, np.full((2, 3), 2.0))


def test_T_backward_restores_the_original_gradient_axes():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
    weights = Tensor([[1, 2], [3, 4], [5, 6]])

    (tensor.T() * weights).backward()

    np.testing.assert_array_equal(tensor.grad, weights.data.T)
    assert tensor.grad.shape == tensor.shape


def test_T_backward_reverses_gradient_axes_for_a_3d_tensor():
    tensor = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)
    weights = Tensor(np.arange(1, 25).reshape(4, 3, 2))

    (tensor.T() * weights).backward()

    np.testing.assert_array_equal(tensor.grad, weights.data.T)
    assert tensor.grad.shape == tensor.shape


def test_backward_through_two_T_operations():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
    weights = Tensor([[1, 2, 3], [4, 5, 6]])

    result = tensor.T().T()
    (result * weights).backward()

    np.testing.assert_array_equal(tensor.grad, weights.data)


def test_repeated_T_backward_accumulates_leaf_gradients():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
    result = tensor.T()

    result.backward()
    result.backward()

    np.testing.assert_array_equal(tensor.grad, np.full((2, 3), 2.0))


def test_getitem_backward_scatters_a_scalar_gradient():
    tensor = Tensor([10, 20, 30, 40], requires_grad=True)

    tensor[2].backward()

    np.testing.assert_array_equal(tensor.grad, [0, 0, 1, 0])
    assert tensor.grad.shape == tensor.shape


def test_getitem_backward_scatters_a_slice_gradient():
    tensor = Tensor(np.arange(12).reshape(3, 4), requires_grad=True)
    weights = Tensor([[1, 2], [3, 4], [5, 6]])

    (tensor[:, 1:3] * weights).backward()

    np.testing.assert_array_equal(
        tensor.grad,
        [[0, 1, 2, 0], [0, 3, 4, 0], [0, 5, 6, 0]],
    )


def test_getitem_backward_supports_multidimensional_integer_indices():
    tensor = Tensor(np.arange(12).reshape(3, 4), requires_grad=True)

    tensor[1, 2].backward()

    expected = np.zeros((3, 4))
    expected[1, 2] = 1
    np.testing.assert_array_equal(tensor.grad, expected)


def test_getitem_backward_scatters_boolean_mask_gradients():
    tensor = Tensor([10, 20, 30, 40], requires_grad=True)
    mask = np.array([True, False, True, False])
    weights = Tensor([2, 5])

    (tensor[mask] * weights).backward()

    np.testing.assert_array_equal(tensor.grad, [2, 0, 5, 0])


def test_getitem_backward_accumulates_repeated_fancy_indices():
    tensor = Tensor([10, 20, 30, 40], requires_grad=True)
    weights = Tensor([1, 2, 3])

    (tensor[[0, 0, 2]] * weights).backward()

    np.testing.assert_array_equal(tensor.grad, [3, 0, 3, 0])


def test_backward_through_multiple_getitem_operations():
    tensor = Tensor(np.arange(10), requires_grad=True)
    weights = Tensor([2, 3])

    result = tensor[2:8][[1, 4]]
    (result * weights).backward()

    np.testing.assert_array_equal(tensor.grad, [0, 0, 0, 2, 0, 0, 3, 0, 0, 0])


def test_repeated_getitem_backward_accumulates_leaf_gradients():
    tensor = Tensor([10, 20, 30, 40], requires_grad=True)
    result = tensor[1:3]

    result.backward()
    result.backward()

    np.testing.assert_array_equal(tensor.grad, [0, 2, 2, 0])


def test_sum_backward_over_all_elements():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)

    tensor.sum().backward()

    np.testing.assert_array_equal(tensor.grad, np.ones((2, 3)))
    assert tensor.grad.shape == tensor.shape


def test_sum_backward_along_a_dimension():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)

    tensor.sum(dim=1).backward()

    np.testing.assert_array_equal(tensor.grad, np.ones((2, 3)))


def test_sum_backward_broadcasts_upstream_gradient():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
    weights = Tensor([2, 3, 4])

    (tensor.sum(dim=0) * weights).backward()

    np.testing.assert_array_equal(tensor.grad, [[2, 3, 4], [2, 3, 4]])


def test_sum_backward_supports_multiple_dimensions_and_keepdims():
    tensor = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)
    weights = Tensor([[[2], [3], [4]]])

    (tensor.sum(dim=(0, 2), keepdims=True) * weights).backward()

    expected = np.broadcast_to(np.array([2, 3, 4]).reshape(1, 3, 1), tensor.shape)
    np.testing.assert_array_equal(tensor.grad, expected)


def test_mean_backward_over_all_elements():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)

    tensor.mean().backward()

    np.testing.assert_allclose(tensor.grad, np.full((2, 3), 1 / 6))


def test_mean_backward_along_a_dimension():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)

    tensor.mean(dim=1).backward()

    np.testing.assert_allclose(tensor.grad, np.full((2, 3), 1 / 3))


def test_mean_backward_broadcasts_and_scales_upstream_gradient():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)
    weights = Tensor([2, 4, 6])

    (tensor.mean(dim=0) * weights).backward()

    np.testing.assert_allclose(tensor.grad, [[1, 2, 3], [1, 2, 3]])


def test_mean_backward_supports_multiple_dimensions():
    tensor = Tensor(np.arange(24).reshape(2, 3, 4), requires_grad=True)
    weights = Tensor([1, 2, 3])

    (tensor.mean(dim=(0, 2)) * weights).backward()

    expected = np.broadcast_to(
        np.array([1, 2, 3]).reshape(1, 3, 1) / 8,
        tensor.shape,
    )
    np.testing.assert_allclose(tensor.grad, expected)


def test_repeated_sum_and_mean_backward_accumulate_leaf_gradients():
    tensor = Tensor(np.arange(6), requires_grad=True)
    summed = tensor.sum()
    averaged = tensor.mean()

    summed.backward()
    averaged.backward()

    np.testing.assert_allclose(tensor.grad, np.full(6, 1 + 1 / 6))


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("dim", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum_and_mean_backward_for_dimension_and_keepdims_combinations(
    reduction,
    dim,
    keepdims,
):
    data = np.arange(1, 13).reshape(3, 4)
    tensor = Tensor(data, requires_grad=True)

    getattr(tensor, reduction)(dim=dim, keepdims=keepdims).backward()

    reduced_element_count = data.size if dim is None else data.shape[dim]
    scale = 1 if reduction == "sum" else 1 / reduced_element_count
    np.testing.assert_allclose(tensor.grad, np.full(data.shape, scale))


def test_var_backward_over_all_elements():
    tensor = Tensor([1, 2, 3, 4], requires_grad=True)

    tensor.var().backward()

    np.testing.assert_allclose(tensor.grad, [-1, -1 / 3, 1 / 3, 1])


def test_var_backward_along_a_dimension():
    tensor = Tensor([[1, 2, 3], [4, 6, 8]], requires_grad=True)

    tensor.var(dim=0).backward()

    np.testing.assert_allclose(
        tensor.grad,
        [[-3, -4, -5], [3, 4, 5]],
    )


def test_var_backward_broadcasts_upstream_gradient():
    tensor = Tensor([[1, 2, 3], [4, 6, 8]], requires_grad=True)
    weights = Tensor([2, 3, 4])

    (tensor.var(dim=0) * weights).backward()

    np.testing.assert_allclose(
        tensor.grad,
        [[-6, -12, -20], [6, 12, 20]],
    )


def test_var_backward_supports_keepdims():
    tensor = Tensor([[1, 2, 3], [4, 6, 8]], requires_grad=True)

    tensor.var(dim=1, keepdims=True).backward()

    expected = 2 * (tensor.data - tensor.data.mean(axis=1, keepdims=True)) / 2
    np.testing.assert_allclose(tensor.grad, expected)


def test_std_backward_over_all_elements():
    tensor = Tensor([1, 2, 3, 4], requires_grad=True)

    tensor.std().backward()

    expected = (tensor.data - tensor.data.mean()) / (
        3 * tensor.data.std(ddof=1)
    )
    np.testing.assert_allclose(tensor.grad, expected)


def test_std_backward_along_a_dimension():
    tensor = Tensor([[1, 2, 3], [4, 6, 8]], requires_grad=True)

    tensor.std(dim=0).backward()

    np.testing.assert_allclose(
        tensor.grad,
        [[-1 / np.sqrt(2)] * 3, [1 / np.sqrt(2)] * 3],
    )


def test_std_backward_broadcasts_upstream_gradient():
    tensor = Tensor([[1, 2, 3], [4, 6, 8]], requires_grad=True)
    weights = Tensor([2, 3, 4])

    (tensor.std(dim=0) * weights).backward()

    np.testing.assert_allclose(
        tensor.grad,
        [
            [-np.sqrt(2), -3 / np.sqrt(2), -2 * np.sqrt(2)],
            [np.sqrt(2), 3 / np.sqrt(2), 2 * np.sqrt(2)],
        ],
    )


def test_std_backward_through_a_constant_column_is_nan():
    tensor = Tensor([[5, 1], [5, 2], [5, 3]], requires_grad=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        tensor.std(dim=0).backward()

    assert np.isnan(tensor.grad[:, 0]).all()
    np.testing.assert_allclose(
        tensor.grad[:, 1],
        [-0.5, 0, 0.5],
    )
