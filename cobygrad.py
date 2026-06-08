import numpy as np

DEFAULT_DTYPE = np.float32


class Tensor:
    def __init__(self, data, _children=(), requires_grad=True, dtype=None):
        self.data = np.array(data, dtype=dtype if dtype is not None else DEFAULT_DTYPE)
        self.grad = np.zeros_like(self.data)
        self.shape = self.data.shape
        self.size = self.data.size
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        out = Tensor(self.data + other.data, _children=(self, other))

        def _backward():
            self.accumulate_grad(lambda: self.unbroadcast(out.grad, self.data.shape))
            other.accumulate_grad(lambda: self.unbroadcast(out.grad, other.data.shape))

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        out = Tensor(self.data * other.data, _children=(self, other))

        def _backward():
            self.accumulate_grad(
                lambda: self.unbroadcast(other.data * out.grad, self.data.shape)
            )
            other.accumulate_grad(
                lambda: self.unbroadcast(self.data * out.grad, other.data.shape)
            )

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return -1.0 * self

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        out = Tensor(self.data / other.data, _children=(self, other))

        def _backward():
            self.accumulate_grad(
                lambda: self.unbroadcast((1 / other.data) * out.grad, self.data.shape)
            )
            other.accumulate_grad(
                lambda: self.unbroadcast(
                    (self.data * (-1 / (other.data**2))) * out.grad, other.data.shape
                )
            )

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other / self

    def __pow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        if other.requires_grad:
            assert np.all(self.data > 0)

        out = Tensor(self.data**other.data, _children=(self, other))

        def _backward():
            self.accumulate_grad(
                lambda: self.unbroadcast(
                    (other.data * self.data ** (other.data - 1)) * out.grad,
                    self.data.shape,
                )
            )
            other.accumulate_grad(
                lambda: self.unbroadcast(
                    (np.log(self.data) * self.data**other.data) * out.grad,
                    other.data.shape,
                )
            )

        out._backward = _backward
        return out

    def __rpow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other**self

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        if self.data.ndim == 1:
            a = self[np.newaxis, :]
            idx1 = 0
        else:
            a = self
            idx1 = slice(None)

        if other.data.ndim == 1:
            b = other[:, np.newaxis]
            idx2 = 0
        else:
            b = other
            idx2 = slice(None)
        return self.matmul_no_vecs(a, b)[..., idx1, idx2]

    def __rmatmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other @ self

    def __getitem__(self, indices):
        out = Tensor(self.data[indices], _children=(self,))

        def _backward():
            zeros = np.zeros_like(self.grad)
            zeros[indices] = out.grad
            self.accumulate_grad(lambda: zeros)

        out._backward = _backward
        return out

    def matmul_no_vecs(self, a, b):
        if not isinstance(a, Tensor):
            a = Tensor(a, requires_grad=False)
        if not isinstance(b, Tensor):
            b = Tensor(b, requires_grad=False)
        out = Tensor(a.data @ b.data, _children=(a, b))

        def _backward():
            a.accumulate_grad(
                lambda: self.unbroadcast(
                    out.grad @ b.data.swapaxes(-1, -2), a.data.shape
                )
            )
            b.accumulate_grad(
                lambda: self.unbroadcast(
                    a.data.swapaxes(-1, -2) @ out.grad, b.data.shape
                )
            )

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis, keepdims=keepdims), _children=(self,))

        def _backward():
            if axis is not None and not keepdims:
                self.accumulate_grad(
                    lambda: np.ones_like(self.data) * np.expand_dims(out.grad, axis)
                )
            else:
                self.accumulate_grad(lambda: np.ones_like(self.data) * out.grad)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        return self.sum(axis, keepdims) / Tensor(
            self.data.shape[axis] if axis is not None else self.data.size
        )

    def var(self, axis=None, keepdims=False):
        return ((self - self.mean(axis, keepdims=True)) ** 2).mean(axis, keepdims)

    def std(self, axis=None, keepdims=False):
        return self.var(axis, keepdims) ** 0.5

    def exp(self):
        out = Tensor(np.exp(self.data), _children=(self,))

        def _backward():
            self.accumulate_grad(lambda: np.exp(self.data) * out.grad)

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), _children=(self,))

        def _backward():
            self.accumulate_grad(lambda: (1 / self.data) * out.grad)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), _children=(self,))

        def _backward():
            self.accumulate_grad(
                lambda: (out.data > 0).astype(out.data.dtype) * out.grad
            )

        out._backward = _backward
        return out

    def sigmoid(self):
        return 1 / (1 + (-self).exp())

    def tanh(self):
        e_x = self.exp()
        e_neg_x = (-self).exp()
        return (e_x - e_neg_x) / (e_x + e_neg_x)

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), _children=(self,))

        def _backward():
            self.accumulate_grad(lambda: out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def concat(self, others, axis=-1):
        tensors = [self, *others]
        out = Tensor(
            np.concatenate([t.data for t in tensors], axis=axis),
            _children=tuple(tensors),
        )

        def _backward():
            sizes = np.cumsum([t.data.shape[axis] for t in tensors])[:-1]
            for t, g in zip(tensors, np.split(out.grad, sizes, axis=axis)):
                t.accumulate_grad(lambda: g)

        out._backward = _backward
        return out

    def swapaxes(self, ax1=-1, ax2=-2):
        out = Tensor(self.data.swapaxes(ax1, ax2), _children=(self,))

        def _backward():
            self.accumulate_grad(lambda: out.grad.swapaxes(ax1, ax2))

        out._backward = _backward
        return out

    def softmax(self, axis=-1, keepdims=True):
        c = Tensor(np.max(self.data, axis, keepdims=True))
        denom = (self - c).exp().sum(axis, keepdims)
        return (self - c).exp() / denom

    def log_softmax(self, axis=-1, keepdims=True):
        return self - self.logsumexp(axis, keepdims)

    def logsumexp(self, axis=-1, keepdims=True):
        c = Tensor(np.max(self.data, axis, keepdims=True))
        return c + (self - c).exp().sum(axis, keepdims).log()

    def unbroadcast(self, array, shape):
        out = np.array(array)
        while out.ndim > len(shape):
            out = out.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                out = out.sum(axis=i, keepdims=True)
        return out

    def backward(self):
        self.grad = np.ones_like(self.data)
        sorted_nodes = []
        visited = set()
        self.visit(sorted_nodes, visited)
        sorted_nodes.reverse()
        for node in sorted_nodes:
            node._backward()

    def visit(self, sorted_nodes, visited):
        if self in visited:
            return
        for child in self._prev:
            child.visit(sorted_nodes, visited)
        visited.add(self)
        sorted_nodes.append(self)

    def accumulate_grad(self, grad):
        if self.requires_grad:
            self.grad += grad()

    def zero_grad(self):
        self.grad = np.zeros_like(self.grad)
        for child in self._prev:
            child.zero_grad()
