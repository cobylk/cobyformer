from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    def __init__(self, param_groups: list[dict]):
        self.param_groups = param_groups
        self.st = {}
        for group in self.param_groups:
            assert group.get("params") is not None
            group["params"] = list(group["params"])
            group.setdefault("lr", 1e-3)

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for group in self.param_groups:
            for param in group["params"]:
                param.grad.fill(0)


class SGD(Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups)
        self.st["vel"] = {}  # keyed by param
        for group in self.param_groups:
            group.setdefault("mu", 0.9)
            group.setdefault("decay", 0.0)

    def step(self):
        for group in self.param_groups:
            for param in group["params"]:
                if self.st["vel"].get(param) is None:
                    self.st["vel"][param] = np.zeros_like(param.data)
                grad = param.grad + group["decay"] * param.data
                self.st["vel"][param] = group["mu"] * self.st["vel"][param] + grad
                param.data -= group["lr"] * self.st["vel"][param]


class Adam(Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups)
        self.st["mean"] = {}
        self.st["var"] = {}
        self.st["step"] = 0
        for group in self.param_groups:
            group.setdefault("b1", 0.9)
            group.setdefault("b2", 0.999)
            group.setdefault("eps", 1e-8)
            group.setdefault("decay", 0.0)

    def step(self):
        self.st["step"] += 1
        for group in self.param_groups:
            for param in group["params"]:
                if self.st["mean"].get(param) is None:
                    self.st["mean"][param] = np.zeros_like(param.data)
                if self.st["var"].get(param) is None:
                    self.st["var"][param] = np.zeros_like(param.data)
                grad = param.grad + group["decay"] * param.data
                self.st["mean"][param] = (
                    group["b1"] * self.st["mean"][param] + (1 - group["b1"]) * grad
                )
                self.st["var"][param] = (
                    group["b2"] * self.st["var"][param] + (1 - group["b2"]) * grad**2
                )

                mean_unbiased = self.st["mean"][param] / (
                    1 - group["b1"] ** self.st["step"]
                )
                variance_unbiased = self.st["var"][param] / (
                    1 - group["b2"] ** self.st["step"]
                )

                param.data -= group["lr"] * (
                    mean_unbiased / (np.sqrt(variance_unbiased) + group["eps"])
                )


class AdamW(Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups)
        self.st["mean"] = {}
        self.st["var"] = {}
        self.st["step"] = 0
        for group in self.param_groups:
            group.setdefault("b1", 0.9)
            group.setdefault("b2", 0.999)
            group.setdefault("eps", 1e-8)
            group.setdefault("decay", 0.01)

    def step(self):
        self.st["step"] += 1
        for group in self.param_groups:
            for param in group["params"]:
                if self.st["mean"].get(param) is None:
                    self.st["mean"][param] = np.zeros_like(param.data)
                if self.st["var"].get(param) is None:
                    self.st["var"][param] = np.zeros_like(param.data)
                self.st["mean"][param] = (
                    group["b1"] * self.st["mean"][param]
                    + (1 - group["b1"]) * param.grad
                )
                self.st["var"][param] = (
                    group["b2"] * self.st["var"][param]
                    + (1 - group["b2"]) * param.grad**2
                )

                mean_unbiased = self.st["mean"][param] / (
                    1 - group["b1"] ** self.st["step"]
                )
                variance_unbiased = self.st["var"][param] / (
                    1 - group["b2"] ** self.st["step"]
                )

                param.data -= group["lr"] * (
                    mean_unbiased / (np.sqrt(variance_unbiased) + group["eps"])
                    + group["decay"] * param.data
                )


class Scheduler(ABC):
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.t = -1

    @abstractmethod
    def step(self):
        pass


class CosineWithLinearWarmup(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps,
        lr_max=1e-3,
        lr_min=1e-4,
        warmup_ratio=0.1,
    ):
        assert 0 <= warmup_ratio < 1
        super().__init__(optimizer)
        self.lr_max, self.lr_min = lr_max, lr_min
        self.total_steps = total_steps
        self.warmup_steps = int(warmup_ratio * total_steps)
        self.step()

    def step(self):
        self.t += 1
        if self.t < self.warmup_steps:
            lr_t = self.lr_max * (self.t + 1) / self.warmup_steps
        else:
            cos_t = min(self.t, self.total_steps) - self.warmup_steps
            lr_t = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + np.cos(np.pi * cos_t / (self.total_steps - self.warmup_steps))
            )
        for group in self.optimizer.param_groups:
            group["lr"] = lr_t
