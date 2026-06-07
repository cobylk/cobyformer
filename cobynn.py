import functools

import numpy as np

from cobygrad import Tensor
from cobyutils import he, xavier


class Parameter(Tensor):
    def __init__(self, data, _children=(), requires_grad=True, type="weight"):
        super().__init__(data, _children=_children, requires_grad=requires_grad)
        self.type = type


class Module:
    _parameters: dict[str, Parameter]
    _modules: dict

    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        if isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def parameters(self):
        yield from self._parameters.values()
        for module in self._modules.values():
            yield from module.parameters()

    def __delattr__(self, name):
        self._parameters.pop(name, None)
        self._modules.pop(name, None)
        object.__delattr__(self, name)

    def forward(self, x: Tensor):
        raise NotImplementedError

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        return self.forward(x)


class Linear(Module):
    def __init__(self, d_in: int, d_out, resid_proj=False, N_layers=1, bias=True):
        super().__init__()
        if resid_proj:
            self.weight = Parameter(he(d_in, d_out) / (np.sqrt(2 * N_layers)))
        else:
            self.weight = Parameter(he(d_in, d_out))
        self.bias = Parameter(np.zeros(d_out), type="bias", requires_grad=bias)

    def forward(self, x: Tensor):
        return x @ self.weight + self.bias


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return x.relu()


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return x.reshape(x.shape[0], -1)


class LayerNorm(Module):
    def __init__(self, d: int, eps=1e-8):
        super().__init__()
        self.gamma = Parameter(np.ones(d), type="norm")
        self.beta = Parameter(np.zeros(d), type="norm")
        self.eps = eps

    def forward(self, x: Tensor):
        x_hat = (x - x.mean(axis=-1, keepdims=True)) / (
            x.var(axis=-1, keepdims=True) + self.eps
        ) ** 0.5
        return self.gamma * x_hat + self.beta


class MultiHeadSelfAttention(Module):
    def __init__(
        self, d: int, h: int, N_layers: int, use_rope=True
    ):  # N_layers for init scaling
        super().__init__()
        assert d % h == 0
        self.d, self.h = d, h
        self.W_k = Parameter(xavier(d, d))
        self.W_q = Parameter(xavier(d, d))
        self.W_v = Parameter(xavier(d, d))
        self.W_o = Parameter(xavier(d, d) / np.sqrt(2 * N_layers))
        self.use_rope = use_rope
        self.sin = {}  # indexed by t
        self.cos = {}

    def forward(self, x: Tensor):
        b, t, d = x.shape
        K, Q, V = (x @ self.W_k, x @ self.W_q, x @ self.W_v)
        K, Q, V = (
            z.reshape(b, t, self.h, d // self.h).swapaxes(1, 2) for z in (K, Q, V)
        )
        # (b, t, d) -> (b, h, t, d/h)
        if self.use_rope:
            if self.cos.get(t) is None or self.sin.get(t) is None:
                self.cos[t], self.sin[t] = self.rope_tables(t, d // self.h)
            Q = self.apply_rope(Q, self.cos[t], self.sin[t])
            K = self.apply_rope(K, self.cos[t], self.sin[t])
        attn = self.dot_self_attention(Q, K, V)  # (b, h, t, d/h)
        attn = attn.swapaxes(1, 2).reshape(b, t, d)
        return attn @ self.W_o

    def dot_self_attention(self, Q, K, V, mask=True):
        d_k, t = K.shape[-1], K.shape[-2]
        scores = Q @ K.swapaxes(-1, -2) / (d_k) ** 0.5
        if mask:
            causal = np.where(np.triu(np.ones((t, t)), k=1), -np.inf, 0.0)
            scores += Tensor(causal, requires_grad=False)
        return scores.softmax(-1) @ V

    def rope_tables(self, t, d_head, base=10000.0):
        inv_freq = base ** (-np.arange(0, d_head, 2) / d_head)  # (d_head/2,)
        angles = np.outer(np.arange(t), inv_freq)  # (t, d_head/2)
        emb = np.concatenate([angles, angles], axis=-1)  # (t, d_head)
        return np.cos(emb), np.sin(emb)

    def rotate_half(self, x):  # x: (b, h, t, d_head)
        half = x.shape[-1] // 2
        return (-x[..., half:]).concat([x[..., :half]], axis=-1)

    def apply_rope(self, x, cos, sin):
        return x * cos + self.rotate_half(x) * sin


class Sequential(Module):
    def __init__(self, *args: Module):
        super().__init__()
        self.sequence = list(args)
        for i, module in enumerate(args):
            self.__setattr__(f"{type(module).__name__}{i}", module)

    def forward(self, x):
        return functools.reduce(lambda val, f: f(val), self.sequence, x)


class MLP(Sequential):
    def __init__(self, d_in: int, d_out: int, d_hidden: int, n_hidden: int):
        super().__init__(
            Flatten(),
            Linear(d_in, d_hidden),
            ReLU(),
            *[
                m
                for _ in range(n_hidden - 1)
                for m in (Linear(d_hidden, d_hidden), ReLU())
            ],
            Linear(d_hidden, d_out),
        )


class CobyformerLayer(Module):
    def __init__(self, d_model, d_ffn, h, use_rope, N_layers):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, h, N_layers, use_rope)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.FFN = Sequential(
            Linear(d_model, d_ffn),
            ReLU(),
            Linear(d_ffn, d_model, N_layers=N_layers, resid_proj=True),
        )

    def forward(self, x: Tensor):
        x += self.attn(self.norm1(x))
        x += self.FFN(self.norm2(x))
        return x


class Cobyformer(Sequential):
    def __init__(self, d_model, d_ffn, h, l_vocab, N_layers):
        super().__init__(
            Linear(l_vocab, d_model, bias=False),
            *[
                CobyformerLayer(d_model, d_ffn, h, True, N_layers)
                for _ in range(N_layers)
            ],
            LayerNorm(d_model),
            Linear(d_model, l_vocab, bias=False),
        )
