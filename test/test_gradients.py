import math

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