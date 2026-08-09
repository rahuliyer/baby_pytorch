from baby_pytorch.value import Value

def test_value_data():
    val = Value(data=42, op='+', label="val1")

    assert val.data == 42
    assert val.op == '+'
    assert val.label == "val1"

def test_value_op():
    val1 = Value(10)
    val2 = Value(20)

    val = Value(42, [val1, val2], op='+')

    assert val.data == 42
    assert val.op == '+'
    assert len(val.children) == 2
    assert val.children == [val1, val2]

def test_value_label():
    val = Value(data=42, label="val1")

    assert val.data == 42
    assert val.label == "val1"

def test_value_no_label():
    val = Value(data=42)

    assert val.data == 42
    assert len(val.label) != 0
    assert "val_" in val.label

def test_repr():
    val = Value(42, label="val")

    assert val.__repr__() == "<Value(data: 42 label: \"val\">"

def test_add():
    val1 = Value(10)
    val2 = Value(20)

    val = val1 + val2

    assert val.data == 30
    assert val.op == '+'
    assert val.children == [val1, val2]


def test_add_constant():
    val1 = Value(10)

    val = 20 + val1

    assert val.data == 30
    assert val.op == '+'
    assert len(val.children) == 2
    assert val1 in val.children

def test_sub():
    val1 = Value(10)
    val2 = Value(20)

    val = val1 - val2

    assert val.data == -10
    assert val.op == '-'
    assert val.children == [val1, val2]

def test_sub_constant():
    val1 = Value(10)

    val = 20 - val1

    assert val.data == -10 
    assert val.op == '-'
    assert len(val.children) == 2
    assert val1 in val.children


def test_mul():
    val1 = Value(10)
    val2 = Value(20)

    val = val1 * val2

    assert val.data == 200
    assert val.op == '*'
    assert val.children == [val1, val2]


def test_rmul():
    val1 = Value(10)

    val = 20 * val1

    assert val.data == 200
    assert val.op == '*'
    assert len(val.children) == 2
    assert val1 in val.children

def test_pow():
    val1 = Value(10)

    val = val1 ** 2

    assert val.data == 100
    assert val.op == '**'
    assert len(val.children) == 1

def test_negative_pow():
    val1 = Value(2)

    val = val1 ** -1

    assert val.data == 0.5

def test_neg():
    val1 = Value(10)

    val = -val1

    assert val.data == -10
    assert val.op == '*'

def test_div():
    val1 = Value(10)
    val2 = Value(2)

    val = val1 / val2

    assert val.data == 5

def test_requires_grad():
    val1 = Value(10)
    val2 = Value(20, requires_grad=False)

    assert val1.requires_grad == True
    assert val2.requires_grad == False
