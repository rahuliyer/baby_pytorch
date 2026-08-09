from baby_pytorch.value import Value

def test_value_data():
    val1 = Value(10)
    val2 = val1 ** 2

    val3 = Value(20)
    val4 = val2 + val3

    val5 = Value(121)
    val6 = val4 - val5

    assert val6.data == -1

    assert val4 in val6.children
    assert val5 in val6.children

    assert val2 in val4.children
    assert val3 in val4.children

    assert val1 in val2.children

def test_value_children():
    val1 = Value(10)

    val = val1 * val1

    assert val.data == 100
    assert len(val.children) == 2
    assert val.children[0] == val1
    assert val.children[1] == val1
