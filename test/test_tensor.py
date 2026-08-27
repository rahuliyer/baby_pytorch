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
