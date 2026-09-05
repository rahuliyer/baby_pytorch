# Baby PyTorch

Baby PyTorch is a small, NumPy-backed neural-network library built to make
automatic differentiation and common deep-learning components easy to inspect.
It follows familiar PyTorch conventions without trying to be a drop-in
replacement for PyTorch.

The project includes reverse-mode autograd, array-aware tensors, neural-network
layers, losses, optimization, initialization, and end-to-end examples. Its test
suite compares values, gradients, layers, and training behavior with PyTorch.

## Features

- NumPy-backed `Tensor` objects with reverse-mode automatic differentiation
- Broadcasting, matrix multiplication, indexing, reshaping, reductions,
  variance, logarithms, and exponentials
- Gradient-free execution with the `no_grad` context manager or decorator
- `Linear`, `Embedding`, `BatchNorm1d`, `Relu`, `Sigmoid`, and `Tanh` modules
- Composable `MLP` models and module weight save/load helpers
- Mean squared error and numerically stable cross-entropy losses
- Stochastic gradient descent with gradient clearing
- Kaiming normal initialization for ReLU, sigmoid, and tanh networks
- PyTorch parity tests for forward values, gradients, and network training

## Setup

The project requires Python 3.14 and uses [uv](https://docs.astral.sh/uv/) for
environment and dependency management.

```bash
git clone https://github.com/rahuliyer/baby_pytorch.git
cd baby_pytorch
uv sync --dev
```

NumPy is the library's numerical backend. Jupyter and Matplotlib support the
examples, while PyTorch is a development-only dependency used by parity tests.

## Automatic differentiation

Create tensors with `requires_grad=True`, build a scalar result, and call
`backward()`:

```python
from baby_pytorch import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = ((2 * x + 1) ** 2).mean()

y.backward()

print(y.data)  # 27.666...
print(x.grad)  # [4.0, 6.666..., 9.333...]
```

Operations build a dynamic computation graph. Gradients accumulate on leaf
tensors across backward calls, while intermediate gradients are reset for each
backward pass.

## Training a model

The module, loss, and optimizer APIs are intentionally close to their PyTorch
counterparts:

```python
import numpy as np

from baby_pytorch import Tensor, no_grad
from baby_pytorch.loss import cross_entropy
from baby_pytorch.nn import MLP, Tanh, kaiming_normal_
from baby_pytorch.optim import SGD

np.random.seed(0)

inputs = Tensor([
    [-1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
])
targets = np.array([0, 1, 1, 0])

model = MLP(2, [8], 2, Tanh())
for layer in model.layers[:-1]:
    kaiming_normal_(layer.weights, layer.weights.shape[0], "tanh")

optimizer = SGD(model.parameters(), lr=0.05)

for _ in range(200):
    optimizer.zero_grad()
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()

with no_grad():
    logits = model(inputs, training=False)
    predictions = logits.data.argmax(axis=1)

print(predictions)
```

Pass `training=False` when stateful modules such as `BatchNorm1d` should use
inference behavior. Wrap inference in `no_grad()` to prevent construction of an
autograd graph.

## Examples

The [`examples`](examples) directory contains two executed notebooks:

- [`example.ipynb`](examples/example.ipynb) trains an MLP on a synthetic
  nonlinear classification problem and visualizes its predictions.
- [`makemore_v3.ipynb`](examples/makemore_v3.ipynb) ports the local Makemore v3
  character-level language model to Baby PyTorch and NumPy. It covers data
  preparation, embeddings, Kaiming initialization, BatchNorm, checkpointing,
  evaluation, and training diagnostics using the committed names dataset.

Launch Jupyter from the repository root:

```bash
uv run jupyter lab
```

## Project layout

```text
src/baby_pytorch/
├── tensor.py                 # Tensor operations and backward entry point
├── no_grad.py                # Gradient-mode context manager
├── functions/                # Graph traversal and gradient propagation
├── nn/                       # Modules, activations, and initialization
├── loss/                     # MSE and cross-entropy losses
└── optim/                    # Optimizer base class and SGD

test/                         # Unit and PyTorch parity tests
examples/                     # Executed training notebooks and local data
```

## Development

Run the complete test suite with:

```bash
uv run pytest -q
```

Baby PyTorch is an educational implementation. It favors readable mechanics
and focused behavior over the performance, hardware support, and breadth of a
production framework.

## Origins: `v1` and micrograd

Baby PyTorch began as a small, scalar-valued autograd engine inspired by
Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). That
original implementation is preserved in the [`v1`](../../tree/v1) tag. Each
`Value` represented one scalar in a dynamically constructed computation graph,
and reverse-mode autodifferentiation walked that graph in topological order.

The snapshot also included scalar `Neuron`, `Layer`, and `MLP` primitives,
activation functions, mean squared error, SGD, and a notebook that trained a
network to classify points inside or outside a circle. The current library grew
from that foundation into the NumPy-backed `Tensor` implementation and the more
PyTorch-like modules documented above.

To explore the original version without moving a branch, check out the tag in
detached-HEAD mode:

```bash
git switch --detach v1
# Return to the current version when finished:
git switch main
```
