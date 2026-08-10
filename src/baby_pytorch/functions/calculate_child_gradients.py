import math


def calculate_child_gradients(val):
    match val.op:
        case '+':
            if val.children[0].requires_grad:
                val.children[0].grad += val.grad

            if val.children[1].requires_grad:
                val.children[1].grad += val.grad
        case '-':
            if val.children[0].requires_grad:
                val.children[0].grad += val.grad

            if val.children[1].requires_grad:
                val.children[1].grad -= val.grad
        case '*':
            if val.children[0].requires_grad:
                val.children[0].grad += val.grad * val.children[1].data

            if val.children[1].requires_grad:
                val.children[1].grad += val.grad * val.children[0].data
        case '**':
            if val.children[0].requires_grad:
                val.children[0].grad += val.grad * val.children[1].data * (
                            val.children[0].data ** (val.children[1].data - 1)
                    )
            if val.children[1].requires_grad:
                val.children[1].grad += (
                    val.grad * val.data * math.log(val.children[0].data)
                )
        case 'tanh':
            if val.children[0].requires_grad:
                val.children[0].grad += val.grad * (1 - val.data ** 2)
        case 'sigmoid':
            if val.children[0].requires_grad:
                val.children[0].grad += val.grad * (val.data * (1 - val.data))
        case 'relu':
            if val.children[0].requires_grad:
                val.children[0].grad += val.grad if val.data > 0 else 0
        case '':
            pass
        case _:
            raise ValueError(f"Unsupported op: {val.op}")
            
            
