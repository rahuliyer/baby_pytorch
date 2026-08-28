from contextlib import contextmanager


@contextmanager
def evaluating(model):
    previous_modes = [(module, module.training) for module in model.modules()]

    try:
        model.eval()
        yield
    finally:
        for module, training in previous_modes:
            module.train(training)
