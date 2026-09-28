"""
Domain Alignment and Reality Gap Adaptation Losses
==================================================
Aligns latent feature representations between structured (Waymo) and unstructured (IDD-PeD) domains:
1. Multi-Kernel Maximum Mean Discrepancy (MK-MMD)
2. Gradient Reversal Layer (GRL) & Domain Adversarial Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """Classifies whether latent biomechanical features originate from Waymo or IDD-PeD."""
    def __init__(self, in_features: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        rev_x = grad_reverse(x, alpha)
        return self.net(rev_x).squeeze(-1)


class MultiKernelMMDLoss(nn.Module):
    """
    Multi-Scale Gaussian RBF Kernel Maximum Mean Discrepancy (MK-MMD).
    Measures distribution discrepancy in RKHS.
    """
    def __init__(self, kernel_muls: list = None, kernel_nums: int = 5):
        super().__init__()
        self.kernel_num = kernel_nums
        self.kernel_mul = 2.0

    def gaussian_kernel(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        l2_distance = ((total0 - total1) ** 2).sum(2)

        bandwidth = torch.sum(l2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        kernel_val = [torch.exp(-l2_distance / (bw + 1e-6)) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return torch.clamp(loss, min=0.0)
