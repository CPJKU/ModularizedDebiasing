from enum import Enum, auto
import contextlib
import torch
from  torch import nn
from torch.nn.parameter import Parameter

from src.utils import concrete_stretched


class DiffWeightFinetune(nn.Module):

    def __init__(
        self,
        weight: nn.Parameter,
        alpha_init: float,
        concrete_lower: float,
        concrete_upper: float,
        structured: bool
    ):
        super().__init__()
        self.concrete_lower = concrete_lower
        self.concrete_upper = concrete_upper
        self.structured = structured

        self.register_parameter("finetune", Parameter(torch.clone(weight)))
        self.register_parameter("alpha", Parameter(torch.zeros_like(weight) + alpha_init))

        if structured:
            self.register_parameter("alpha_group", Parameter(torch.zeros((1,), device=weight.device) + alpha_init))

        self.active = True

    def forward(self, X):
        if not self.active: return X
        diff = (self.finetune - X).detach()
        return (self.finetune - diff) + self.diff_weight(X)

    def diff_weight(self, X):
        return self.z * (self.finetune - X)

    @property
    def z(self) -> Parameter:
        z = self.dist(self.alpha)
        if self.structured:
            z *= self.dist(self.alpha_group)
        return z

    @property
    def alpha_weights(self) -> list:
        alpha = [self.alpha]
        if self.structured:
            alpha.append(self.alpha_group)
        return alpha

    def dist(self, alpha) -> torch.Tensor:
        return concrete_stretched(
            alpha,
            l=self.concrete_lower,
            r=self.concrete_upper,
            deterministic=(not self.training)
        )

    def set_frozen(self, frozen: bool) -> None:
        self.finetune.requires_grad = not frozen
        self.alpha.requires_grad = not frozen
        if self.structured:
            self.alpha_group.requires_grad = not frozen
        if frozen:
            self.eval()
        else:
            self.train()


class DiffWeightFixmask(nn.Module):

    def __init__(self, diff_weight: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.register_parameter("diff_weight", Parameter(diff_weight * mask))
        self.register_parameter("mask", Parameter(mask, requires_grad=False))
        self.active = True

    def forward(self, X):
        if not self.active: return X
        return X + self.mask * self.diff_weight

    def set_frozen(self, frozen: bool) -> None:
        self.diff_weight.requires_grad = not frozen