from contextlib import ContextDecorator
from contextvars import ContextVar

_GRAD_ENABLED = ContextVar("grad_enabled", default=True)


class no_grad(ContextDecorator):
    def __init__(self):
        self._tokens = []

    def __enter__(self):
        self._tokens.append(_GRAD_ENABLED.set(False))

        return self

    def __exit__(self, exc_type, exc, tb):
        _GRAD_ENABLED.reset(self._tokens.pop())

        return False

    def _recreate_cm(self):
        return self.__class__()


def grad_enabled():
    return _GRAD_ENABLED.get()
