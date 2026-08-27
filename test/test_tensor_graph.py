import sys

import pytest

from baby_pytorch.tensor import Tensor

def test_tensor_graph():
    t1 = Tensor(10)
    t2 = t1 ** 2

    t3 = Tensor(20)
    t4 = t2 + t3

    t5 = Tensor(121)
    t6 = t4 - t5

    assert t6.data == -1

    assert t4 in t6.children
    assert t5 in t6.children

    assert t2 in t4.children
    assert t3 in t4.children

    assert t1 in t2.children

def test_tensor_repeated_children():
    t1 = Tensor(10)

    t = t1 * t1

    assert t.data == 100
    assert len(t.children) == 2
    assert t.children[0] == t1
    assert t.children[1] == t1

def test_tensor_grad():
    t1 = Tensor(10, requires_grad=True)
    t2 = t1 * 2

    assert t2.data == 20
    assert len(t2.children) == 2

    for child in t2.children:
        if child == t1:
            assert child.requires_grad == True
        else:
            assert child.requires_grad == False


def test_topo_sort_handles_deep_graphs():
    # A graph deeper than the interpreter's recursion limit; a recursive
    # topological sort would blow the stack here.
    t = Tensor(1.0, requires_grad=True)

    out = t
    for _ in range(sys.getrecursionlimit() * 10):
        out = out + 1

    out.backward()

    assert t.grad == pytest.approx(1.0)
