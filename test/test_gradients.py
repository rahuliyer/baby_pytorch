import math

import pytest

from baby_pytorch.value import Value
from baby_pytorch.activation_functions import tanh, sigmoid, relu


def test_chain_rule_for_all_binary_operations():
    x = Value(2.0)
    y = 3 * ((x + 1) * (x - 4))

    y.backward()

    assert x.grad == pytest.approx(3.0)


def test_power_gradients_for_base_and_exponent():
    base = Value(2.0)
    exponent = Value(3.0)

    (4 * (base**exponent)).backward()

    assert base.grad == pytest.approx(48.0)
    assert exponent.grad == pytest.approx(32.0 * math.log(2.0))


def test_shared_nodes_are_processed_once():
    x = Value(3.0)
    square = x * x

    (square + square).backward()

    assert x.grad == pytest.approx(12.0)


def test_scalar_division_and_reverse_arithmetic():
    x = Value(2.0)

    y = (10 - x) + (12 / x) + (x / 2)
    y.backward()

    assert y.data == pytest.approx(15.0)
    assert x.grad == pytest.approx(-3.5)


def test_result_does_not_require_grad_when_operands_do_not():
    left = Value(2.0, requires_grad=False)
    right = Value(3.0, requires_grad=False)

    assert not (left + right).requires_grad
    assert not (left * right).requires_grad
    assert not (left**right).requires_grad


def test_repeated_backward_accumulates_leaf_gradients_linearly():
    x = Value(2.0)
    y = (x * 2) * 3

    y.backward()
    y.backward()

    assert x.grad == pytest.approx(12.0)

def test_activation_function_backward():
    for v, f, expected in [(2.0, tanh, 0.7065082485316443),
                    (2.0, sigmoid, 1.0499358540350662),
                    (2.0, relu, 10.0),
                    (-2.0, relu, 0)]:
        x = Value(v)
        (10 * f(x)).backward()
        assert x.grad == pytest.approx(expected)