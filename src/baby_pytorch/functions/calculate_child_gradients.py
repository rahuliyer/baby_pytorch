import numpy as np

from .unbroadcast import unbroadcast

def calculate_child_gradients(tensor):
    match tensor.op:
        case '+':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += unbroadcast(
                    tensor.grad,
                    tensor.children[0].grad.shape
                ) 

            if tensor.children[1].requires_grad:
                tensor.children[1].grad += unbroadcast(
                    tensor.grad,
                    tensor.children[1].grad.shape
                )
        case '-':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += unbroadcast(
                    tensor.grad,
                    tensor.children[0].grad.shape
                )

            if tensor.children[1].requires_grad:
                tensor.children[1].grad -= unbroadcast(
                    tensor.grad,
                    tensor.children[1].grad.shape
                )
        case '*':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += unbroadcast(
                    tensor.grad * tensor.children[1].data,
                    tensor.children[0].grad.shape
                )

            if tensor.children[1].requires_grad:
                tensor.children[1].grad += unbroadcast(
                    tensor.grad * tensor.children[0].data,
                    tensor.children[1].grad.shape
                )
        case '**':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += unbroadcast(
                    tensor.grad * tensor.children[1].data * (
                            tensor.children[0].data ** (tensor.children[1].data - 1)
                    ),
                    tensor.children[0].grad.shape
                )
            if tensor.children[1].requires_grad:
                tensor.children[1].grad += unbroadcast(
                    (tensor.grad * tensor.data * np.log(tensor.children[0].data)),
                     tensor.children[1].grad.shape                
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

    assert tensor.data.shape == tensor.grad.shape
            
