import torch
from torch import Tensor


def robust_l1_loss(pred_mean: Tensor, pred_log_std: Tensor, target: Tensor) -> Tensor:
    """Robust L1 loss using a Lorentzian prior. Trains the model to learn to predict
    aleatoric (i.e. per-sample) uncertainty.

    Args:
        pred_mean (Tensor): Tensor of predicted means.
        pred_log_std (Tensor): Tensor of predicted log standard deviations representing
            per-sample model uncertainties.
        target (Tensor): Tensor of target values.

    Returns:
        Tensor: Evaluated robust L1 loss
    """
    loss = 2**0.5 * (pred_mean - target).abs() * torch.exp(-pred_log_std) + pred_log_std
    return torch.mean(loss)


def robust_l2_loss(pred_mean: Tensor, pred_log_std: Tensor, target: Tensor) -> Tensor:
    """Robust L2 loss using a Gaussian prior. Trains the model to learn to predict
    aleatoric (i.e. per-sample) uncertainty.

    Args:
        pred_mean (Tensor): Tensor of predicted means.
        pred_log_std (Tensor): Tensor of predicted log standard deviations representing
            per-sample model uncertainties.
        target (Tensor): Tensor of target values.

    Returns:
        Tensor: Evaluated robust L2 loss
    """
    loss = 0.5 * (pred_mean - target) ** 2 * torch.exp(-2 * pred_log_std) + pred_log_std
    return torch.mean(loss)


# aliases for backwards compatibility
RobustL1Loss = robust_l1_loss
RobustL2Loss = robust_l2_loss
