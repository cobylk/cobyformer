# cobyformer

This repo has an autograd engine, transformer, and some basic pytorch-y OOP setup that I've written by hand for the purposes of (1) improving at numpy and Python OOP and (2) strongly internalizing how ML frameworks work so that I can have extremely accurate mental models of my code when doing research.

I typed all of the code in here with my fingers and without the use of LLMs (that would defeat the purpose). I did use LLMs for some conceptual help, asked questions about syntax, and I asked for architectural/algorithmic feedback about implementations only after first getting them to work correctly myself.

Lastly, I used an LLM to generate the following elucidation of my code, because I am lazy:


#### `cobygrad.py` — the autograd engine

This is the foundation everything else sits on. It defines a single `Tensor` class that wraps a NumPy array (`self.data`) and carries a gradient of the same shape (`self.grad`), a `requires_grad` flag, a `_backward` closure, and a set of parent tensors (`_prev`) that together form the computation graph. Each operation builds a new tensor whose `_backward` closure knows how to push the incoming gradient back to its inputs.

The arithmetic dunders (`__add__`, `__mul__`, `__sub__`, `__truediv__`, `__pow__`, `__matmul__`, and their reflected `__r*__` variants) and `__getitem__` each construct an output tensor and define the corresponding local gradient rule. Broadcasting is handled in reverse by `unbroadcast`, which sums a gradient back down to the shape of the operand it belongs to, so the forward pass can broadcast freely without breaking the backward pass. Matrix multiplication is split out into `matmul_no_vecs` with a wrapper in `__matmul__` that promotes 1-D operands to matrices and squeezes the result back, so vectors and batched/stacked matrices all work through one code path.

On top of the operators there are reductions (`sum`, `mean`, `var`, `std`), elementwise functions (`exp`, `log`, `relu`, `sigmoid`, `tanh`), shape manipulation (`reshape`, `concat`, `swapaxes`), and numerically stable `softmax`, `log_softmax`, and `logsumexp` (each subtracts the max before exponentiating). The reverse pass lives in `backward`: it seeds the output gradient with ones, builds a reverse topological order of the graph via the recursive `visit`, and then calls each node's `_backward` in turn. `accumulate_grad` gates accumulation on `requires_grad`, and `zero_grad` recurses through the graph to reset gradients.

#### `cobynn.py` — the PyTorch-style module system and transformer

This file reproduces the ergonomics of `torch.nn` on top of the autograd engine. `Parameter` subclasses `Tensor` and adds a `type` tag (weight, bias, norm). `Module` is the base class: it intercepts `__setattr__` so that any `Parameter` or sub-`Module` assigned as an attribute is automatically registered in `_parameters` or `_modules`, and `parameters()` walks that structure recursively. Subclasses implement `forward`, and `__call__` wraps inputs into tensors before dispatching to it.

The concrete layers are `Linear` (He-initialized weights with an optional residual-projection scaling of 1/√(2·N_layers) and an optionally trainable bias), `ReLU`, `Flatten`, and `LayerNorm` (with learnable `gamma`/`beta`). `MultiHeadSelfAttention` implements multi-head causal self-attention with Xavier-initialized Q/K/V/O projections, an output projection scaled for residual stability, scaled-dot-product attention with a causal mask built from `triu`, and rotary position embeddings (RoPE) whose sin/cos tables are cached per sequence length. `Sequential` chains modules with a `functools.reduce` over the call, and `MLP` is a `Sequential` that builds a flatten-then-stacked-linear-ReLU network.

The transformer itself is two classes. `CobyformerLayer` is a pre-norm block: it normalizes, attends, and adds the residual, then normalizes, runs a two-layer feed-forward network, and adds again. `Cobyformer` is a `Sequential` stack of an input projection from vocab to model dimension, `N_layers` of `CobyformerLayer`, a final `LayerNorm`, and an output projection back to vocabulary logits.

#### `cobyoptim.py` — optimizers and learning-rate schedules

This module follows PyTorch's param-group convention: optimizers take a list of dicts, each with a `params` list and optional per-group hyperparameters that are filled in with `setdefault`. `Optimizer` is an ABC providing `zero_grad` and an abstract `step`. `SGD` implements momentum with optional weight decay. `Adam` and `AdamW` both keep first- and second-moment estimates with bias correction; the difference is that `AdamW` applies decoupled weight decay directly to the parameters rather than folding it into the gradient. `Scheduler` is a second ABC, and `CosineWithLinearWarmup` ramps the learning rate linearly during a warmup fraction and then follows a cosine decay from `lr_max` down to `lr_min`.

#### `cobyutils.py` — losses and initializers

The helpers shared across training scripts. `cross_entropy_loss` works from probabilities, while `cross_entropy_loss_with_logits` applies `log_softmax` internally and one-hot-encodes the targets, which is the numerically sensible path; `CrossEntropyLoss` is a thin callable wrapper around the latter. `he` and `xavier` return normally distributed weight matrices scaled by fan-in (and fan-out, for Xavier).

#### `cobymnist.py` — an end-to-end training script

A working demonstration that ties the pieces together. It builds an `MLP`, an `AdamW` optimizer, and a cosine-with-warmup schedule, fetches MNIST through `sklearn.datasets.fetch_openml`, and runs a manual train/validation loop in minibatches of 100, printing loss and accuracy each epoch. This is the proof that the autograd engine, modules, optimizers, and losses actually compose into something that learns.

#### `cobychar.py` and `cobytok.py` — placeholders

Both files are currently empty. The names suggest the intended next steps: a character-level training script to exercise the `Cobyformer`, and a tokenizer to feed it.

#### Project files

`pyproject.toml` carries the project metadata (Python ≥ 3.11) and configures the `ty` type checker; `uv.lock` is the resolved dependency lockfile for `uv`; `.gitignore` excludes the virtual environment, caches, and editor state.
