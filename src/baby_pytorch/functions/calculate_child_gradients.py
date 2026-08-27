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
        case 'log':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += (
                    tensor.grad / tensor.children[0].data
                )
        case 'log10':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += (
                    tensor.grad / (tensor.children[0].data * np.log(10))
                )
        case 'exp':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += tensor.grad * tensor.data
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
        case 'matmul':
            c1, c2 = tensor.children
            if not c1.requires_grad and not c2.requires_grad:
                return
            
            c1_is_vector = c1.data.ndim == 1
            c2_is_vector = c2.data.ndim == 1
            grad = tensor.grad

            if c1_is_vector and c2_is_vector:
                # the grad is a scalar
                if c1.requires_grad:
                    c1.grad += np.expand_dims(grad, 0) @ np.expand_dims(c2.data, -1).T

                if c2.requires_grad:
                    c2.grad += np.expand_dims(c1.data, 0).T @ np.expand_dims(grad, -1)
            elif c1_is_vector and not c2_is_vector:
                # grad is a matrix with the -2 dimension removed
                if c1.requires_grad:
                    grad_c1 = (
                        np.expand_dims(grad, -2)
                        @ c2.data.swapaxes(-1, -2)
                    ).squeeze(-2)
                    c1.grad += unbroadcast(grad_c1, c1.grad.shape)

                if c2.requires_grad:
                    grad_c2 = (
                        np.expand_dims(c1.data, -1)
                        @ np.expand_dims(grad, -2)
                    )
                    c2.grad += unbroadcast(grad_c2, c2.grad.shape)
            elif c2_is_vector and not c1_is_vector:
                if c1.requires_grad:
                    grad_c1 = (
                        np.expand_dims(grad, -1)
                        @ np.expand_dims(c2.data, -2)
                    )
                    c1.grad += unbroadcast(grad_c1, c1.grad.shape)

                if c2.requires_grad:
                    grad_c2 = (
                        c1.data.swapaxes(-1, -2)
                        @ np.expand_dims(grad, -1)
                    ).squeeze(-1)
                    c2.grad += unbroadcast(grad_c2, c2.grad.shape)
            else:
                c1_t = c1.data.swapaxes(-1, -2)
                c2_t = c2.data.swapaxes(-1, -2)

                if c1.requires_grad:
                    c1.grad += unbroadcast(tensor.grad @ c2_t, c1.grad.shape)

                if c2.requires_grad:
                    c2.grad += unbroadcast(c1_t @ tensor.grad, c2.grad.shape)
        case 'reshape':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += tensor.grad.reshape(
                    tensor.ctx["original_shape"]
                )
        case 'swapaxes':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += tensor.grad.swapaxes(
                    tensor.ctx["axis1"], 
                    tensor.ctx["axis2"]
                )
        case 'T':
            if tensor.children[0].requires_grad:
                tensor.children[0].grad += tensor.grad.T
        case '':
            pass
        case _:
            raise ValueError(f"Unsupported op: {tensor.op}")

    assert tensor.data.shape == tensor.grad.shape
            
