import math


def calculate_child_gradients(tensor):
    match tensor.op:
        case '+':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += tensor.grad

            if tensor.children[1].requires_grad:
                tensor.children[1].grad += tensor.grad
        case '-':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += tensor.grad

            if tensor.children[1].requires_grad:
                tensor.children[1].grad -= tensor.grad
        case '*':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += tensor.grad * tensor.children[1].data

            if tensor.children[1].requires_grad:
                tensor.children[1].grad += tensor.grad * tensor.children[0].data
        case '**':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += tensor.grad * tensor.children[1].data * (
                            tensor.children[0].data ** (tensor.children[1].data - 1)
                    )
            if tensor.children[1].requires_grad:
                tensor.children[1].grad += (
                    tensor.grad * tensor.data * math.log(tensor.children[0].data)
                )
        case 'tanh':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += tensor.grad * (1 - tensor.data ** 2)
        case 'sigmoid':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += (
                    tensor.grad * (tensor.data * (1 - tensor.data))
                )
        case 'relu':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += tensor.grad if tensor.data > 0 else 0
        case '':
            pass
        case _:
            raise ValueError(f"Unsupported op: {tensor.op}")
            
            
