from baby_pytorch.value import Value

def test_value_graph():
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

def test_value_repeated_children():
    val1 = Value(10)

    val = val1 * val1

    assert val.data == 100
    assert len(val.children) == 2
    assert val.children[0] == val1
    assert val.children[1] == val1

def test_value_grad():
    val1 = Value(10)
    val2 = val1 * 2

    assert val2.data == 20
    assert len(val2.children) == 2

    for child in val2.children:
        if child == val1:
            assert child.requires_grad == True
        else:
            assert child.requires_grad == False
