"""End-to-end parity between baby_pytorch and PyTorch networks.

`test_pytorch_parity.py` compares a single hand-built expression graph.  This
module goes one level up: it builds equivalent *networks* in both frameworks,
trains them with SGD, and asserts that the losses, the gradients of every
parameter, and the parameters themselves stay identical at every step.

Both frameworks run in float64 (baby_pytorch tensors are float64 by default),
so the comparisons can be tight.  The initial weights always come from
baby_pytorch and are copied into the PyTorch model, which keeps the two in
lockstep without relying on the two RNGs agreeing.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from baby_pytorch.loss import MSE, cross_entropy
from baby_pytorch.nn import (
    MLP,
    BatchNorm1d,
    Embedding,
    Linear,
    Relu,
    Sigmoid,
    Tanh,
)
from baby_pytorch.optim import SGD
from baby_pytorch.tensor import Tensor

INPUT_SIZE = 6
HIDDEN_LAYERS = [8, 5]
OUTPUT_SIZE = 4
BATCH_SIZE = 7
TRAINING_STEPS = 10
LEARNING_RATE = 0.1

# Both frameworks are in float64 and start from bit-identical weights, so the
# only difference is floating point operation ordering.
TOLERANCE = {"rtol": 1e-9, "atol": 1e-11}

ACTIVATIONS = {
    "tanh": (Tanh, torch.nn.Tanh),
    "relu": (Relu, torch.nn.ReLU),
    "sigmoid": (Sigmoid, torch.nn.Sigmoid),
}


def _assert_close(actual, expected, description):
    np.testing.assert_allclose(
        actual,
        expected,
        err_msg=f"mismatch in {description}",
        **TOLERANCE,
    )


def _torch_linears(torch_net):
    return [
        module
        for module in torch_net
        if isinstance(module, torch.nn.Linear)
    ]


def _copy_weights_into_torch(baby_layer, torch_layer):
    # baby_pytorch stores weights as (fan_in, fan_out); torch uses
    # (fan_out, fan_in), so every crossing of that boundary transposes.
    with torch.no_grad():
        torch_layer.weight.copy_(torch.tensor(baby_layer.weights.data.T))
        torch_layer.bias.copy_(torch.tensor(baby_layer.bias.data))


def _build_mlp_pair(activation="tanh", seed=0):
    """Build a baby_pytorch MLP and a PyTorch model with the same weights."""
    baby_activation, torch_activation = ACTIVATIONS[activation]

    np.random.seed(seed)
    baby_net = MLP(
        INPUT_SIZE,
        HIDDEN_LAYERS,
        OUTPUT_SIZE,
        baby_activation(),
    )

    layer_sizes = [INPUT_SIZE, *HIDDEN_LAYERS, OUTPUT_SIZE]
    modules = []
    for index, (fan_in, fan_out) in enumerate(
        zip(layer_sizes, layer_sizes[1:])
    ):
        modules.append(torch.nn.Linear(fan_in, fan_out))
        # The final layer produces logits, so it has no activation.
        if index < len(layer_sizes) - 2:
            modules.append(torch_activation())
    torch_net = torch.nn.Sequential(*modules).double()

    for baby_layer, torch_layer in zip(
        baby_net.layers,
        _torch_linears(torch_net),
    ):
        _copy_weights_into_torch(baby_layer, torch_layer)

    return baby_net, torch_net


def _paired_parameters(baby_net, torch_net):
    """Yield (name, baby parameter, torch parameter, transposed) tuples."""
    for index, (baby_layer, torch_layer) in enumerate(
        zip(baby_net.layers, _torch_linears(torch_net))
    ):
        yield f"layer {index} weights", baby_layer.weights, torch_layer.weight, True
        yield f"layer {index} bias", baby_layer.bias, torch_layer.bias, False


def _assert_parameters_match(pairs, description):
    for name, baby_parameter, torch_parameter, transposed in pairs:
        expected = torch_parameter.detach().numpy()
        if transposed:
            expected = expected.T

        assert baby_parameter.shape == expected.shape, (
            f"shape mismatch in {name} value {description}: "
            f"{baby_parameter.shape} != {expected.shape}"
        )
        _assert_close(baby_parameter.data, expected, f"{name} value {description}")


def _assert_gradients_match(pairs, description):
    for name, baby_parameter, torch_parameter, transposed in pairs:
        assert torch_parameter.grad is not None, (
            f"torch did not produce a gradient for {name} {description}"
        )
        expected = torch_parameter.grad.numpy()
        if transposed:
            expected = expected.T

        assert baby_parameter.grad.shape == expected.shape, (
            f"shape mismatch in {name} gradient {description}: "
            f"{baby_parameter.grad.shape} != {expected.shape}"
        )
        _assert_close(
            baby_parameter.grad,
            expected,
            f"{name} gradient {description}",
        )


def _train_and_compare(pairs, batches, baby_loss_fn, torch_loss_fn):
    """Train both networks step for step, comparing everything as we go."""
    baby_optimizer = SGD(
        [baby_parameter for _, baby_parameter, _, _ in pairs],
        lr=LEARNING_RATE,
    )
    torch_optimizer = torch.optim.SGD(
        [torch_parameter for _, _, torch_parameter, _ in pairs],
        lr=LEARNING_RATE,
    )

    # A network that never moves would pass every comparison below.
    initial_weights = pairs[0][1].data.copy()

    _assert_parameters_match(pairs, "before training")

    for step, batch in enumerate(batches):
        baby_optimizer.zero_grad()
        torch_optimizer.zero_grad()

        baby_loss = baby_loss_fn(batch)
        torch_loss = torch_loss_fn(batch)
        _assert_close(
            baby_loss.data,
            torch_loss.detach().numpy(),
            f"loss at step {step}",
        )

        baby_loss.backward()
        torch_loss.backward()
        _assert_gradients_match(pairs, f"at step {step}")

        baby_optimizer.step()
        torch_optimizer.step()
        _assert_parameters_match(pairs, f"after step {step}")

    assert not np.allclose(pairs[0][1].data, initial_weights), (
        "training did not change the weights, so the comparison is vacuous"
    )


def _classification_batches(steps=TRAINING_STEPS, seed=1234):
    rng = np.random.default_rng(seed)
    return [
        (
            rng.normal(size=(BATCH_SIZE, INPUT_SIZE)),
            rng.integers(0, OUTPUT_SIZE, size=BATCH_SIZE),
        )
        for _ in range(steps)
    ]


def _regression_batches(steps=TRAINING_STEPS, seed=99):
    rng = np.random.default_rng(seed)
    return [
        (
            rng.normal(size=(BATCH_SIZE, INPUT_SIZE)),
            rng.normal(size=(BATCH_SIZE, OUTPUT_SIZE)),
        )
        for _ in range(steps)
    ]


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_cross_entropy_training_matches_pytorch(activation):
    baby_net, torch_net = _build_mlp_pair(activation=activation)

    def baby_loss_fn(batch):
        inputs, targets = batch
        return cross_entropy(baby_net(Tensor(inputs)), targets)

    def torch_loss_fn(batch):
        inputs, targets = batch
        return F.cross_entropy(
            torch_net(torch.tensor(inputs, dtype=torch.float64)),
            torch.tensor(targets, dtype=torch.long),
        )

    _train_and_compare(
        list(_paired_parameters(baby_net, torch_net)),
        _classification_batches(),
        baby_loss_fn,
        torch_loss_fn,
    )


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_mse_training_matches_pytorch(activation):
    baby_net, torch_net = _build_mlp_pair(activation=activation)

    def baby_loss_fn(batch):
        inputs, targets = batch
        return MSE(baby_net(Tensor(inputs)), Tensor(targets))

    def torch_loss_fn(batch):
        inputs, targets = batch
        return F.mse_loss(
            torch_net(torch.tensor(inputs, dtype=torch.float64)),
            torch.tensor(targets, dtype=torch.float64),
        )

    _train_and_compare(
        list(_paired_parameters(baby_net, torch_net)),
        _regression_batches(),
        baby_loss_fn,
        torch_loss_fn,
    )


def test_repeated_batches_match_pytorch():
    """Re-running the same batch checks that zero_grad() agrees with torch."""
    baby_net, torch_net = _build_mlp_pair()
    inputs, targets = _classification_batches(steps=1)[0]
    batches = [(inputs, targets)] * TRAINING_STEPS

    def baby_loss_fn(batch):
        return cross_entropy(baby_net(Tensor(batch[0])), batch[1])

    def torch_loss_fn(batch):
        return F.cross_entropy(
            torch_net(torch.tensor(batch[0], dtype=torch.float64)),
            torch.tensor(batch[1], dtype=torch.long),
        )

    _train_and_compare(
        list(_paired_parameters(baby_net, torch_net)),
        batches,
        baby_loss_fn,
        torch_loss_fn,
    )


def test_evaluation_after_training_matches_pytorch():
    baby_net, torch_net = _build_mlp_pair()

    def baby_loss_fn(batch):
        return cross_entropy(baby_net(Tensor(batch[0])), batch[1])

    def torch_loss_fn(batch):
        return F.cross_entropy(
            torch_net(torch.tensor(batch[0], dtype=torch.float64)),
            torch.tensor(batch[1], dtype=torch.long),
        )

    _train_and_compare(
        list(_paired_parameters(baby_net, torch_net)),
        _classification_batches(),
        baby_loss_fn,
        torch_loss_fn,
    )

    held_out = np.random.default_rng(7).normal(size=(3, INPUT_SIZE))
    baby_predictions = baby_net(Tensor(held_out), training=False)

    torch_net.eval()
    with torch.no_grad():
        torch_predictions = torch_net(
            torch.tensor(held_out, dtype=torch.float64)
        )

    _assert_close(
        baby_predictions.data,
        torch_predictions.numpy(),
        "predictions in evaluation mode",
    )
    assert not baby_predictions.requires_grad


VOCABULARY_SIZE = 10
EMBEDDING_DIMS = 4
CONTEXT_SIZE = 3


def _build_embedding_pair(seed=0):
    """An embedding table feeding a linear layer, in both frameworks."""
    np.random.seed(seed)
    baby_embedding = Embedding(VOCABULARY_SIZE, EMBEDDING_DIMS)
    baby_linear = Linear(CONTEXT_SIZE * EMBEDDING_DIMS, OUTPUT_SIZE)

    torch_embedding = torch.nn.Embedding(
        VOCABULARY_SIZE,
        EMBEDDING_DIMS,
    ).double()
    torch_linear = torch.nn.Linear(
        CONTEXT_SIZE * EMBEDDING_DIMS,
        OUTPUT_SIZE,
    ).double()

    with torch.no_grad():
        torch_embedding.weight.copy_(
            torch.tensor(baby_embedding.embedding.data)
        )
    _copy_weights_into_torch(baby_linear, torch_linear)

    pairs = [
        ("embedding", baby_embedding.embedding, torch_embedding.weight, False),
        ("linear weights", baby_linear.weights, torch_linear.weight, True),
        ("linear bias", baby_linear.bias, torch_linear.bias, False),
    ]

    return (baby_embedding, baby_linear), (torch_embedding, torch_linear), pairs


def test_embedding_network_training_matches_pytorch():
    """Repeated indices exercise gradient accumulation into the table."""
    baby_modules, torch_modules, pairs = _build_embedding_pair()
    baby_embedding, baby_linear = baby_modules
    torch_embedding, torch_linear = torch_modules

    rng = np.random.default_rng(2024)
    batches = []
    for _ in range(TRAINING_STEPS):
        indices = rng.integers(
            0,
            VOCABULARY_SIZE // 2,  # a small range guarantees repeated indices
            size=(BATCH_SIZE, CONTEXT_SIZE),
        )
        targets = rng.integers(0, OUTPUT_SIZE, size=BATCH_SIZE)
        batches.append((indices, targets))

    def baby_loss_fn(batch):
        indices, targets = batch
        embedded = baby_embedding(indices)
        flattened = embedded.reshape(
            BATCH_SIZE,
            CONTEXT_SIZE * EMBEDDING_DIMS,
        )
        return cross_entropy(baby_linear(flattened), targets)

    def torch_loss_fn(batch):
        indices, targets = batch
        embedded = torch_embedding(torch.tensor(indices, dtype=torch.long))
        flattened = embedded.reshape(
            BATCH_SIZE,
            CONTEXT_SIZE * EMBEDDING_DIMS,
        )
        return F.cross_entropy(
            torch_linear(flattened),
            torch.tensor(targets, dtype=torch.long),
        )

    _train_and_compare(pairs, batches, baby_loss_fn, torch_loss_fn)


def _build_batch_norm_pair(seed=0):
    """A Linear -> BatchNorm1d -> Tanh -> Linear network in both frameworks."""
    np.random.seed(seed)
    baby_first = Linear(INPUT_SIZE, HIDDEN_LAYERS[0])
    baby_norm = BatchNorm1d(HIDDEN_LAYERS[0], eps=1e-5, momentum=0.01)
    baby_activation = Tanh()
    baby_second = Linear(HIDDEN_LAYERS[0], OUTPUT_SIZE)

    torch_first = torch.nn.Linear(INPUT_SIZE, HIDDEN_LAYERS[0])
    torch_norm = torch.nn.BatchNorm1d(
        HIDDEN_LAYERS[0],
        eps=1e-5,
        momentum=0.01,
    )
    torch_net = torch.nn.Sequential(
        torch_first,
        torch_norm,
        torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN_LAYERS[0], OUTPUT_SIZE),
    ).double()
    torch_second = torch_net[3]

    _copy_weights_into_torch(baby_first, torch_first)
    _copy_weights_into_torch(baby_second, torch_second)
    with torch.no_grad():
        torch_norm.weight.copy_(torch.tensor(baby_norm.gamma.data))
        torch_norm.bias.copy_(torch.tensor(baby_norm.beta.data))

    def baby_forward(inputs):
        hidden = baby_first(Tensor(inputs))
        hidden = baby_norm(hidden)
        hidden = baby_activation(hidden)
        return baby_second(hidden)

    pairs = [
        ("first weights", baby_first.weights, torch_first.weight, True),
        ("first bias", baby_first.bias, torch_first.bias, False),
        ("gamma", baby_norm.gamma, torch_norm.weight, False),
        ("beta", baby_norm.beta, torch_norm.bias, False),
        ("second weights", baby_second.weights, torch_second.weight, True),
        ("second bias", baby_second.bias, torch_second.bias, False),
    ]

    return baby_forward, torch_net, pairs


def test_batch_norm_network_training_matches_pytorch():
    baby_forward, torch_net, pairs = _build_batch_norm_pair()

    def baby_loss_fn(batch):
        inputs, targets = batch
        return cross_entropy(baby_forward(inputs), targets)

    def torch_loss_fn(batch):
        inputs, targets = batch
        return F.cross_entropy(
            torch_net(torch.tensor(inputs, dtype=torch.float64)),
            torch.tensor(targets, dtype=torch.long),
        )

    _train_and_compare(
        pairs,
        _classification_batches(),
        baby_loss_fn,
        torch_loss_fn,
    )
