from torch import nn
from torch.autograd import Function

from typing import Union, Optional


class GradScaler(Function):
    # https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/2
    @staticmethod
    def forward(ctx, input_, lmbda):
        ctx.lmbda = lmbda
        return input_.view_as(input_)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output * ctx.lmbda
        return grad_input, None


class ClfHead(nn.Module):

    ACTIVATIONS = nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['gelu', nn.GELU()],
        ['tanh', nn.Tanh()]
    ])

    def __init__(
        self,
        hid_sizes: Union[int, list, tuple],
        num_labels: int,
        activation: str = 'tanh',
        dropout: Optional[float] = None
    ):
        super().__init__()

        if isinstance(hid_sizes, int):
            hid_sizes = [hid_sizes]
            out_sizes = [num_labels]
        elif isinstance(hid_sizes, (list, tuple)):
            if len(hid_sizes)==1:
                out_sizes = [num_labels]
            else:
                out_sizes = hid_sizes[1:] + [num_labels]
        else:
            raise ValueError(f"hid_sizes has to be of type int or list but got {type(hid_sizes)}")

        layers = []
        for i, (hid_size, out_size) in enumerate(zip(hid_sizes, out_sizes)):
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.extend([
                nn.Linear(hid_size, out_size),
                self.ACTIVATIONS[activation]
            ])
        layers = layers[:-1] # remove last activation

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def freeze_parameters(self, frozen: bool):
        for p in self.parameters():
            p.requires_grad = not frozen


class AdvHead(nn.Module):
    def __init__(self, adv_count: int = 1, **kwargs):
        super().__init__()
        self.heads = nn.ModuleList()
        for i in range(adv_count):
            self.heads.append(ClfHead(**kwargs))

    def forward(self, x):
        out = []
        for head in self.heads:
            out.append(head(x))
        return out

    def forward_reverse(self, x, lmbda = 1.):
        lmbda *= -1
        x_ = GradScaler.apply(x, lmbda)
        return self(x_)

    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()


class AdvHeadWrapper(nn.Module):
    def __init__(self, adv_heads: Union[list, tuple]):
        super().__init__()
        self.adv_heads = nn.ModuleList(adv_heads)

    def forward(self, x):
        out = []
        for adv_head in self.adv_heads:
            out.extend(adv_head(x))
        return out

    def forward_reverse(self, x, lmbda = 1.):
        out_rev = []
        for adv_head in self.adv_heads:
            out_rev.extend(adv_head.forward_reverse(x, lmbda))
        return out_rev

    def reset_parameters(self):
        for adv_head in self.adv_heads:
            adv_head.reset_parameters()