from contextlib import contextmanager


@contextmanager
def evaluating(model):
    prev_state = model.training

    model.eval()

    try:
        yield
    finally:
        model.train(prev_state)
