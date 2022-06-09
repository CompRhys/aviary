import numpy as np
import torch
from torch import Tensor


def RobustL1Loss(pred_mean: Tensor, pred_log_std: Tensor, target: Tensor) -> Tensor:
    """Robust L1 loss using a Lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.

    Args:
        pred_mean (Tensor): Tensor of predicted means
        pred_log_std (Tensor): Tensor of predicted log standard deviations
        target (Tensor): Tensor of target values

    Returns:
        Tensor: Evaluated robust L1 loss
    """
    loss = (
        np.sqrt(2.0) * torch.abs(pred_mean - target) * torch.exp(-pred_log_std)
        + pred_log_std
    )
    return torch.mean(loss)


def RobustL2Loss(pred_mean: Tensor, pred_log_std: Tensor, target: Tensor) -> Tensor:
    """Robust L2 loss using a Gaussian prior. Allows for estimation
    of an aleatoric uncertainty.

    Args:
        pred_mean (Tensor): Tensor of predicted means
        pred_log_std (Tensor): Tensor of predicted log standard deviations
        target (Tensor): Tensor of target values

    Returns:
        Tensor: Evaluated robust L2 loss
    """
    loss = (
        0.5 * torch.pow(pred_mean - target, 2.0) * torch.exp(-2.0 * pred_log_std)
        + pred_log_std
    )
    return torch.mean(loss)
