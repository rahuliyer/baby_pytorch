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
