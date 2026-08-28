import numpy as np
import pytest

from baby_pytorch.tensor import Tensor

def test_tensor_data():
    t = Tensor(data=42, op='+', label="t1")

    assert t.data == 42
    assert t.op == '+'
    assert t.label == "t1"

def test_tensor_op():
    t1 = Tensor(10)
    t2 = Tensor(20)

    t = Tensor(42, [t1, t2], op='+')

    assert t.data == 42
    assert t.op == '+'
    assert len(t.children) == 2
    assert t.children == [t1, t2]

def test_tensor_label():
    t = Tensor(data=42, label="t1")

    assert t.data == 42
    assert t.label == "t1"

def test_tensor_no_label():
    t = Tensor(data=42)

    assert t.data == 42
    assert len(t.label) != 0
    assert "tensor_" in t.label

def test_repr():
    t = Tensor(42, label="t")

    assert t.__repr__() == "<Tensor(data: 42.0 label: \"t\">"

def test_add():
    t1 = Tensor(10)
    t2 = Tensor(20)

    t = t1 + t2

    assert t.data == 30
    assert t.op == '+'
    assert t.children == [t1, t2]


def test_add_constant():
    t1 = Tensor(10)

    t = 20 + t1

    assert t.data == 30
    assert t.op == '+'
    assert len(t.children) == 2
    assert t1 in t.children

def test_sub():
    t1 = Tensor(10)
    t2 = Tensor(20)

    t = t1 - t2

    assert t.data == -10
    assert t.op == '-'
    assert t.children == [t1, t2]

def test_sub_constant():
    t1 = Tensor(10)

    t = 20 - t1

    assert t.data == 10
    assert t.op == '-'
    assert len(t.children) == 2
    assert t1 in t.children


def test_mul():
    t1 = Tensor(10)
    t2 = Tensor(20)

    t = t1 * t2

    assert t.data == 200
    assert t.op == '*'
    assert t.children == [t1, t2]


def test_rmul():
    t1 = Tensor(10)

    t = 20 * t1

    assert t.data == 200
    assert t.op == '*'
    assert len(t.children) == 2
    assert t1 in t.children

def test_pow():
    t1 = Tensor(10)

    t = t1 ** 2

    assert t.data == 100
    assert t.op == '**'
    assert len(t.children) == 2

def test_negative_pow():
    t1 = Tensor(2)

    t = t1 ** -1

    assert t.data == 0.5

def test_neg():
    t1 = Tensor(10)

    t = -t1

    assert t.data == -10
    assert t.op == '*'

def test_div():
    t1 = Tensor(10)
    t2 = Tensor(2)

    t = t1 / t2

    assert t.data == 5

def test_requires_grad():
    t1 = Tensor(10)
    t2 = Tensor(20, requires_grad=False)
    t3 = Tensor(30, requires_grad=True)

    assert t1.requires_grad == False
    assert t2.requires_grad == False
    assert t3.requires_grad == True


def test_log():
    t = Tensor([1.0, np.e, np.e**2])

    result = t.log()

    np.testing.assert_allclose(result.data, [0.0, 1.0, 2.0])
    assert result.op == 'log'
    assert result.children == [t]


def test_log10():
    t = Tensor([1.0, 10.0, 1000.0])

    result = t.log10()

    np.testing.assert_allclose(result.data, [0.0, 1.0, 3.0])
    assert result.op == 'log10'
    assert result.children == [t]


def test_exp():
    t = Tensor([0.0, 1.0, 2.0])

    result = t.exp()

    np.testing.assert_allclose(result.data, [1.0, np.e, np.e**2])
    assert result.op == 'exp'
    assert result.children == [t]


def test_log_exp_propagate_requires_grad():
    tracked = Tensor(2.0, requires_grad=True)
    plain = Tensor(2.0)

    assert tracked.log().requires_grad
    assert tracked.log10().requires_grad
    assert tracked.exp().requires_grad
    assert not plain.log().requires_grad
    assert not plain.log10().requires_grad
    assert not plain.exp().requires_grad


def test_matmul_vector_dot_product():
    left = Tensor([1, 2, 3])
    right = Tensor([4, 5, 6])

    result = left @ right

    assert result.data == pytest.approx(32)
    assert result.shape == ()


def test_matmul_matrix_by_vector():
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])
    vector = Tensor([1, 2, 3])

    result = matrix @ vector

    np.testing.assert_array_equal(result.data, [14, 32])
    assert result.shape == (2,)


def test_matmul_vector_by_matrix():
    vector = Tensor([1, 2])
    matrix = Tensor([[3, 4, 5], [6, 7, 8]])

    result = vector @ matrix

    np.testing.assert_array_equal(result.data, [15, 18, 21])
    assert result.shape == (3,)


def test_matmul_matrix_by_matrix():
    left = Tensor([[1, 2, 3], [4, 5, 6]])
    right = Tensor([[7, 8], [9, 10], [11, 12]])

    result = left @ right

    np.testing.assert_array_equal(result.data, [[58, 64], [139, 154]])
    assert result.shape == (2, 2)


def test_matmul_supports_batched_matrices():
    left_data = np.arange(12).reshape(2, 2, 3)
    right_data = np.arange(12).reshape(2, 3, 2)
    left = Tensor(left_data)
    right = Tensor(right_data)

    result = left @ right

    np.testing.assert_array_equal(result.data, np.matmul(left_data, right_data))
    assert result.shape == (2, 2, 2)


def test_matmul_broadcasts_batch_dimensions():
    left_data = np.arange(12).reshape(2, 2, 3)
    right_data = np.arange(6).reshape(3, 2)
    left = Tensor(left_data)
    right = Tensor(right_data)

    result = left @ right

    np.testing.assert_array_equal(result.data, np.matmul(left_data, right_data))
    assert result.shape == (2, 2, 2)


def test_matmul_tracks_graph_metadata_and_requires_grad():
    left = Tensor([[1, 2]], requires_grad=False)
    right = Tensor([[3], [4]], requires_grad=True)

    result = left @ right

    assert result.op == "matmul"
    assert result.children == [left, right]
    assert result.requires_grad


def test_matmul_without_tracked_operands_does_not_require_grad():
    left = Tensor([[1, 2]])
    right = Tensor([[3], [4]])

    result = left @ right

    assert not result.requires_grad


def test_matmul_rejects_incompatible_inner_dimensions():
    left = Tensor(np.zeros((2, 3)))
    right = Tensor(np.zeros((2, 4)))

    with pytest.raises(ValueError):
        left @ right


def test_reshape_changes_shape_without_changing_values():
    tensor = Tensor([1, 2, 3, 4, 5, 6])

    result = tensor.reshape(2, 3)

    np.testing.assert_array_equal(result.data, [[1, 2, 3], [4, 5, 6]])
    assert result.shape == (2, 3)
    assert tensor.shape == (6,)


def test_reshape_supports_an_inferred_dimension():
    tensor = Tensor(np.arange(12))

    result = tensor.reshape(3, -1)

    np.testing.assert_array_equal(result.data, np.arange(12).reshape(3, 4))
    assert result.shape == (3, 4)


def test_reshape_accepts_a_shape_tuple():
    tensor = Tensor(np.arange(6))

    result = tensor.reshape((2, 3))

    np.testing.assert_array_equal(result.data, np.arange(6).reshape(2, 3))
    assert result.shape == (2, 3)


def test_reshape_tracks_graph_metadata_and_requires_grad():
    tensor = Tensor(np.arange(6), requires_grad=True)

    result = tensor.reshape(2, 3)

    assert result.op == "reshape"
    assert result.children == [tensor]
    assert result.requires_grad
    assert result.ctx["original_shape"] == (6,)


def test_reshape_of_an_untracked_tensor_does_not_require_grad():
    tensor = Tensor(np.arange(6))

    assert not tensor.reshape(2, 3).requires_grad


def test_reshape_rejects_an_incompatible_shape():
    tensor = Tensor(np.arange(6))

    with pytest.raises(ValueError):
        tensor.reshape(4, 2)


def test_view_changes_shape_without_changing_values():
    tensor = Tensor([1, 2, 3, 4, 5, 6])

    result = tensor.view(2, 3)

    np.testing.assert_array_equal(result.data, [[1, 2, 3], [4, 5, 6]])
    assert result.shape == (2, 3)
    assert tensor.shape == (6,)


def test_view_supports_an_inferred_dimension():
    tensor = Tensor(np.arange(12))

    result = tensor.view(3, -1)

    np.testing.assert_array_equal(result.data, np.arange(12).reshape(3, 4))
    assert result.shape == (3, 4)


def test_view_accepts_a_shape_tuple():
    tensor = Tensor(np.arange(6))

    result = tensor.view((2, 3))

    np.testing.assert_array_equal(result.data, np.arange(6).reshape(2, 3))
    assert result.shape == (2, 3)


def test_view_tracks_graph_metadata_and_requires_grad():
    tensor = Tensor(np.arange(6), requires_grad=True)

    result = tensor.view(2, 3)

    assert result.op == "reshape"
    assert result.children == [tensor]
    assert result.requires_grad
    assert result.ctx["original_shape"] == (6,)


def test_view_of_an_untracked_tensor_does_not_require_grad():
    tensor = Tensor(np.arange(6))

    assert not tensor.view(2, 3).requires_grad


def test_view_rejects_an_incompatible_shape():
    tensor = Tensor(np.arange(6))

    with pytest.raises(ValueError):
        tensor.view(4, 2)


def test_swapaxes_transposes_a_matrix():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = tensor.swapaxes(0, 1)

    np.testing.assert_array_equal(result.data, [[1, 4], [2, 5], [3, 6]])
    assert result.shape == (3, 2)
    assert tensor.shape == (2, 3)


def test_swapaxes_supports_multidimensional_tensors():
    data = np.arange(24).reshape(2, 3, 4)
    tensor = Tensor(data)

    result = tensor.swapaxes(0, 2)

    np.testing.assert_array_equal(result.data, np.swapaxes(data, 0, 2))
    assert result.shape == (4, 3, 2)


def test_swapaxes_supports_negative_axes():
    data = np.arange(24).reshape(2, 3, 4)
    tensor = Tensor(data)

    result = tensor.swapaxes(-1, -2)

    np.testing.assert_array_equal(result.data, np.swapaxes(data, -1, -2))
    assert result.shape == (2, 4, 3)


def test_swapaxes_with_the_same_axis_leaves_values_unchanged():
    data = np.arange(6).reshape(2, 3)
    tensor = Tensor(data)

    result = tensor.swapaxes(1, 1)

    np.testing.assert_array_equal(result.data, data)
    assert result.shape == tensor.shape


def test_swapaxes_tracks_graph_metadata_and_requires_grad():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)

    result = tensor.swapaxes(0, 1)

    assert result.op == "swapaxes"
    assert result.children == [tensor]
    assert result.requires_grad
    assert result.ctx == {"axis1": 0, "axis2": 1}


def test_swapaxes_of_an_untracked_tensor_does_not_require_grad():
    tensor = Tensor(np.arange(6).reshape(2, 3))

    assert not tensor.swapaxes(0, 1).requires_grad


def test_swapaxes_rejects_an_out_of_range_axis():
    tensor = Tensor(np.arange(6).reshape(2, 3))

    with pytest.raises(np.exceptions.AxisError):
        tensor.swapaxes(0, 2)


def test_T_transposes_a_matrix():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = tensor.T()

    np.testing.assert_array_equal(result.data, [[1, 4], [2, 5], [3, 6]])
    assert result.shape == (3, 2)
    assert tensor.shape == (2, 3)


def test_T_leaves_a_vector_unchanged():
    tensor = Tensor([1, 2, 3])

    result = tensor.T()

    np.testing.assert_array_equal(result.data, tensor.data)
    assert result.shape == (3,)


def test_T_reverses_all_axes_of_a_multidimensional_tensor():
    data = np.arange(24).reshape(2, 3, 4)
    tensor = Tensor(data)

    result = tensor.T()

    np.testing.assert_array_equal(result.data, data.T)
    assert result.shape == (4, 3, 2)


def test_T_twice_restores_the_original_values_and_shape():
    data = np.arange(24).reshape(2, 3, 4)
    tensor = Tensor(data)

    result = tensor.T().T()

    np.testing.assert_array_equal(result.data, data)
    assert result.shape == tensor.shape


def test_T_tracks_graph_metadata_and_requires_grad():
    tensor = Tensor(np.arange(6).reshape(2, 3), requires_grad=True)

    result = tensor.T()

    assert result.op == "T"
    assert result.children == [tensor]
    assert result.requires_grad


def test_T_of_an_untracked_tensor_does_not_require_grad():
    tensor = Tensor(np.arange(6).reshape(2, 3))

    assert not tensor.T().requires_grad


def test_getitem_selects_a_single_element():
    tensor = Tensor([10, 20, 30])

    result = tensor[1]

    assert result.data == 20
    assert result.shape == ()


def test_getitem_supports_negative_indices():
    tensor = Tensor([10, 20, 30])

    assert tensor[-1].data == 30


def test_getitem_supports_slices():
    tensor = Tensor([0, 1, 2, 3, 4, 5])

    result = tensor[1:5:2]

    np.testing.assert_array_equal(result.data, [1, 3])
    assert result.shape == (2,)


def test_getitem_supports_multidimensional_indices():
    tensor = Tensor(np.arange(12).reshape(3, 4))

    result = tensor[1:, 1:3]

    np.testing.assert_array_equal(result.data, [[5, 6], [9, 10]])
    assert result.shape == (2, 2)


def test_getitem_supports_ellipsis():
    data = np.arange(24).reshape(2, 3, 4)
    tensor = Tensor(data)

    result = tensor[..., 2]

    np.testing.assert_array_equal(result.data, data[..., 2])
    assert result.shape == (2, 3)


def test_getitem_supports_boolean_masks():
    tensor = Tensor([10, 20, 30, 40])
    mask = np.array([True, False, True, False])

    result = tensor[mask]

    np.testing.assert_array_equal(result.data, [10, 30])


def test_getitem_supports_fancy_integer_indices():
    tensor = Tensor([10, 20, 30, 40])

    result = tensor[[3, 1, 1]]

    np.testing.assert_array_equal(result.data, [40, 20, 20])


def test_getitem_tracks_graph_metadata_and_requires_grad():
    tensor = Tensor(np.arange(6), requires_grad=True)

    result = tensor[2:5]

    assert result.op == "index"
    assert result.children == [tensor]
    assert result.requires_grad
    assert result.ctx["index"] == slice(2, 5)


def test_getitem_of_an_untracked_tensor_does_not_require_grad():
    tensor = Tensor(np.arange(6))

    assert not tensor[2:5].requires_grad


def test_getitem_rejects_an_out_of_range_index():
    tensor = Tensor([10, 20, 30])

    with pytest.raises(IndexError):
        tensor[3]


def test_sum_reduces_all_elements_to_a_scalar():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = tensor.sum()

    assert result.data == pytest.approx(21)
    assert result.shape == ()


def test_sum_reduces_along_a_dimension():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = tensor.sum(dim=0)

    np.testing.assert_array_equal(result.data, [5, 7, 9])
    assert result.shape == (3,)


def test_sum_supports_multiple_and_negative_dimensions():
    data = np.arange(24).reshape(2, 3, 4)
    tensor = Tensor(data)

    multiple = tensor.sum(dim=(0, 2))
    negative = tensor.sum(dim=-1)

    np.testing.assert_array_equal(multiple.data, data.sum(axis=(0, 2)))
    np.testing.assert_array_equal(negative.data, data.sum(axis=-1))


def test_sum_can_keep_reduced_dimensions():
    tensor = Tensor(np.arange(6).reshape(2, 3))

    result = tensor.sum(dim=1, keepdims=True)

    np.testing.assert_array_equal(result.data, [[3], [12]])
    assert result.shape == (2, 1)


def test_sum_tracks_graph_metadata_and_requires_grad():
    tensor = Tensor(np.arange(6), requires_grad=True)

    result = tensor.sum()

    assert result.op == "sum"
    assert result.children == [tensor]
    assert result.requires_grad


def test_sum_of_an_untracked_tensor_does_not_require_grad():
    assert not Tensor([1, 2, 3]).sum().requires_grad


def test_mean_reduces_all_elements_to_a_scalar():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = tensor.mean()

    assert result.data == pytest.approx(3.5)
    assert result.shape == ()


def test_mean_reduces_along_a_dimension():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = tensor.mean(dim=0)

    np.testing.assert_array_equal(result.data, [2.5, 3.5, 4.5])
    assert result.shape == (3,)


def test_mean_supports_multiple_and_negative_dimensions():
    data = np.arange(24).reshape(2, 3, 4)
    tensor = Tensor(data)

    multiple = tensor.mean(dim=(0, 2))
    negative = tensor.mean(dim=-1)

    np.testing.assert_allclose(multiple.data, data.mean(axis=(0, 2)))
    np.testing.assert_allclose(negative.data, data.mean(axis=-1))


def test_mean_can_keep_reduced_dimensions():
    tensor = Tensor(np.arange(6).reshape(2, 3))

    result = tensor.mean(dim=1, keepdims=True)

    np.testing.assert_array_equal(result.data, [[1], [4]])
    assert result.shape == (2, 1)


def test_mean_tracks_graph_metadata_and_requires_grad():
    tensor = Tensor(np.arange(6), requires_grad=True)

    result = tensor.mean()

    assert result.op == "mean"
    assert result.children == [tensor]
    assert result.requires_grad


def test_mean_of_an_untracked_tensor_does_not_require_grad():
    assert not Tensor([1, 2, 3]).mean().requires_grad
