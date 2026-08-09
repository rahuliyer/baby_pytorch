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

