import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelFocalLoss(nn.Module):
    """Focal loss for multi-label classification."""
    def __init__(self, alpha: float | torch.Tensor = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C) raw model outputs (no sigmoid applied)
        targets: (B, C) multi-hot {0, 1} labels
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - p_t).clamp(min=1e-6) ** self.gamma

        if isinstance(self.alpha, torch.Tensor):
            alpha_t = self.alpha.to(logits.device) * targets + (1 - self.alpha.to(logits.device)) * (1 - targets)
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_term * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_alpha_from_freq(label_matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes per-class alpha for focal loss from label frequencies."""
    pos_freq = label_matrix.float().mean(dim=0).clamp(min=eps, max=1 - eps)
    alpha = 1.0 - pos_freq
    return alpha


if __name__ == "__main__":
    # quick sanity check 
    torch.manual_seed(0)
    B, C = 8, 14
    logits = torch.randn(B, C, requires_grad=True)
    targets = (torch.rand(B, C) > 0.85).float()  # sparse positives, like real data

    loss_fn = MultiLabelFocalLoss(alpha=0.25, gamma=2.0)
    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(loss)
    print(f"[ok] focal loss forward+backward, loss={loss.item():.4f}, grad norm={logits.grad.norm().item():.4f}")

    freqs = (torch.rand(2000, C) > 0.9).float()
    alpha = compute_alpha_from_freq(freqs)
    print(f"[ok] per-class alpha from freq, shape={tuple(alpha.shape)}, range=({alpha.min():.3f}, {alpha.max():.3f})")
