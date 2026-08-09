from baby_pytorch.value import Value

def test_value_data():
    val = Value(10) ** 2 + Value(20) - Value(121)

    assert val.data == -1

def test_value_children():
    val1 = Value(10)

    val = val1 * val1

    assert val.data == 100
    assert len(val.children) == 2
    assert val.children[0] == val1
    assert val.children[1] == val1
