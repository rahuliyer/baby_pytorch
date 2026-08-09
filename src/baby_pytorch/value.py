import random
import string


class Value:
    def __init__(self, data, children=[], op='', label=''):
        self.data = data
        self.children = children
        self.op = op

        if label != '':
            self.label = label
        else:
            suffix = ''.join(
                random.choice(string.ascii_lowercase + string.digits)
                for _ in range(4)
            )
            self.label = f'val_{suffix}'

    def __repr__(self):
        return f"<Value(data: {self.data} label: \"{self.label}\">"

    def __add__(self, other):
        other = Value(other) if type(other) is not Value else other

        out = Value(
                data=self.data + other.data,
                children=[self, other],
                op='+'
        )

        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = Value(other) if type(other) is not Value else other

        out = Value(
                data=self.data - other.data,
                children=[self, other],
                op='-'
        )

        return out

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        other = Value(other) if type(other) is not Value else other

        out = Value(
                data=self.data * other.data,
                children=[self, other],
                op='*'
        )

        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        other = Value(other) if type(other) is not Value else other

        out = Value(
                data=self.data ** other.data,
                children=[],
                op='**'
        )

        return out
        
