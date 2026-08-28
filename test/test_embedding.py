import numpy as np
import torch

from baby_pytorch.nn import Embedding
from baby_pytorch.tensor import Tensor


def test_embedding_initializes_a_trainable_parameter():
    embedding = Embedding(5, 3)

    assert embedding.embedding.shape == (5, 3)
    assert embedding.embedding.requires_grad
    assert embedding.parameters() == [embedding.embedding]
    assert repr(embedding) == "Embedding: shape((5, 3))"


def test_embedding_accepts_integer_arrays_and_tensors():
    embedding = Embedding(4, 2)
    embedding.embedding = Tensor(
        [[0.1, 0.2], [1.1, 1.2], [2.1, 2.2], [3.1, 3.2]],
        requires_grad=True,
    )

    array_result = embedding(np.array([3, 1]))
    tensor_result = embedding(Tensor([3, 1], dtype=np.int64))

    expected = [[3.1, 3.2], [1.1, 1.2]]
    np.testing.assert_allclose(array_result.data, expected)
    np.testing.assert_allclose(tensor_result.data, expected)


def test_embedding_values_and_repeated_index_gradients_match_pytorch():
    embedding_data = np.array(
        [[0.2, -0.4], [1.0, 0.5], [-0.3, 0.8], [0.7, -1.2]],
    )
    index_data = np.array([0, 2, 2, 3])
    coefficient_data = np.array(
        [[1.0, 2.0], [0.5, -1.0], [3.0, 0.25], [-2.0, 1.5]],
    )

    embedding = Embedding(4, 2)
    embedding.embedding = Tensor(embedding_data, requires_grad=True)
    index = Tensor(index_data, dtype=np.int64)
    coefficients = Tensor(coefficient_data)

    baby_result = embedding(index)
    (baby_result * coefficients).sum().backward()

    torch_embedding = torch.tensor(
        embedding_data,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch_result = torch_embedding[torch.tensor(index_data)]
    (torch_result * torch.tensor(coefficient_data)).sum().backward()

    np.testing.assert_allclose(baby_result.data, torch_result.detach().numpy())
    np.testing.assert_allclose(
        embedding.embedding.grad,
        torch_embedding.grad.numpy(),
    )


def test_embedding_save_and_load_use_independent_tensor_leaves():
    embedding = Embedding(3, 2)
    embedding.embedding.data[:] = [[1, 2], [3, 4], [5, 6]]

    saved = embedding.save_weights()
    embedding.embedding.data[0, 0] = 99

    assert not saved[0].requires_grad
    assert saved[0].children == []
    assert not np.shares_memory(saved[0].data, embedding.embedding.data)
    assert saved[0].data[0, 0] == 1

    restored = Embedding(3, 2)
    restored.load_weights(saved)

    np.testing.assert_array_equal(restored.embedding.data, saved[0].data)
    assert restored.embedding.requires_grad
    assert restored.embedding.children == []
    assert not np.shares_memory(restored.embedding.data, saved[0].data)
