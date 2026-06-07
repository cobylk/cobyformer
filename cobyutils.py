import numpy as np

from cobygrad import Tensor


def cross_entropy_loss(p: Tensor, y: Tensor):
    if y.size == 1:
        y = Tensor(np.eye(p.data.size)[int(y.data)])
    return -(y * p.log()).sum()


def cross_entropy_loss_with_logits(p: Tensor, y: Tensor):
    if not isinstance(y, Tensor):
        y = Tensor(y)
    if not isinstance(p, Tensor):
        p = Tensor(p)
    p = p.log_softmax()
    y = Tensor(np.eye(p.data.shape[-1])[y.data.astype(int)])
    return -(y * p).sum()


def he(d_in: int, d_out: int):
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0 / d_in), size=(d_in, d_out))


def xavier(d_in: int, d_out: int):
    return np.random.normal(
        loc=0.0, scale=np.sqrt(2.0 / (d_in + d_out)), size=(d_in, d_out)
    )


class CrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(self, p: Tensor, y: Tensor):
        return cross_entropy_loss_with_logits(p, y)
