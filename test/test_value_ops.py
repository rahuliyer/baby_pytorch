from baby_pytorch.value import Value

def test_value_data():
    val = Value(10) ** 2 + Value(20) - Value(121)

    assert val.data == -1
