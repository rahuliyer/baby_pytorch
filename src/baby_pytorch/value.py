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
