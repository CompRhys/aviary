import os
import torch
from tqdm import trange
import shutil
import math
import numpy as np

from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse
from torch.optim.lr_scheduler import _LRScheduler
from sampnn.data import AverageMeter, Normalizer

def evaluate(generator, model, criterion, optimizer, 
            normalizer, device, task="train", verbose=True):
    """ 
    evaluate the model 
    """

    losses = AverageMeter()
    errors = AverageMeter()

    if task == "train":
        model.train()
    elif task == "val":
        model.eval()
    elif task == "test":
        model.eval()
        test_targets = []
        test_pred = []
        test_var = []
        test_cif_ids = []
        test_comp = []
    else:
        raise NameError("Only train, val or test is allowed as task")
    
    with trange(len(generator), disable=(not verbose)) as t:
        for input_, target, batch_comp, batch_cif_ids in generator:
            
            # normalize target
            target_norm = normalizer.norm(target)
            
            # move tensors to GPU
            input_ = (tensor.to(device) for tensor in input_ )
            target_norm = target_norm.to(device)

            # compute output
            # output = model(*input_)
            output, log_var = model(*input_).chunk(2,dim=1)

            # loss = criterion(output, target_norm)
            loss = criterion(output, log_var, target_norm)
            losses.update(loss.data.cpu().item(), target.size(0))

            # measure accuracy and record loss
            pred = normalizer.denorm(output.data.cpu())
            var = normalizer.denorm_var(torch.exp(log_var).data.cpu())
            
            mae_error = mae(pred, target)
            errors.update(mae_error, target.size(0))

            # rmse_error = mse(pred.exp_(), target.exp_()).sqrt_()
            # rmse_error = mse(pred, target).sqrt_()
            # errors.update(rmse_error, target.size(0))

            if task == "test":
                # collect the model outputs
                test_cif_ids += batch_cif_ids
                test_comp += batch_comp
                test_targets += target.view(-1).tolist()
                test_pred += pred.view(-1).tolist()
                test_var += var.view(-1).tolist()
            elif task == "train":
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t.set_postfix(loss=losses.val)
            t.update()


    if task == "test":  
        print("Test : Loss {loss.avg:.4f}\t "
              "Error {error.avg:.3f}\n".format(loss=losses, error=errors))
        return test_cif_ids, test_comp, test_targets, test_pred, test_var
    else:
        return losses.avg, errors.avg


def save_checkpoint(state, is_best, 
                    checkpoint="checkpoint.pth.tar", 
                    best="best.pth.tar" ):
    """
    Saves a checkpoint and overwrites the best model when is_best = True
    """

    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)


def load_previous_state(path, model, optimizer, normalizer):
    """
    """
    assert os.path.isfile(path), "no checkpoint found at '{}'".format(path) 

    checkpoint = torch.load(path)
    start_epoch = checkpoint["epoch"]
    best_error = checkpoint["best_error"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    normalizer.load_state_dict(checkpoint["normalizer"])
    print("Loaded Previous Model '{}' (epoch {})"
            .format(path, checkpoint["epoch"]))

    return model, optimizer, normalizer, best_error, start_epoch


def RobustL1(output, log_var, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation 
    of an aleatoric uncertainty. 
    """
    loss = np.sqrt(2.0) * torch.abs(output - target) * \
           torch.exp(-0.5 * log_var) + 0.5 * log_var
    return torch.mean(loss)


def RobustL2(output, log_var, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation 
    of an aleatoric uncertainty.
    """
    loss = torch.pow(output - target, 2.0) * \
           torch.exp(- log_var) + 0.5 * log_var
    return torch.mean(loss)


class CosineAnnealingRestartsLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule with warm restarts, where :math:`\eta_{max}` is set to the
    initial learning rate, :math:`T_{cur}` is the number of epochs since the
    last restart and :math:`T_i` is the number of epochs in :math:`i`-th run
    (after performing :math:`i` restarts). If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2} \eta_{mult}^i (\eta_{max}-\eta_{min})
        (1 + \cos(\frac{T_{cur}}{T_i - 1}\pi))
        T_i = T T_{mult}^i
    Notice that because the schedule is defined recursively, the learning rate
    can be simultaneously modified outside this scheduler by other operators.
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that in the
    paper the :math:`i`-th run takes :math:`T_i + 1` epochs, while in this
    implementation it takes :math:`T_i` epochs only. This implementation
    also enables updating the range of learning rates by multiplicative factor
    :math:`\eta_{mult}` after each restart.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T (int): Length of the initial run (in number of epochs).
        eta_min (float): Minimum learning rate. Default: 0.
        T_mult (float): Multiplicative factor adjusting number of epochs in
            the next run that is applied after each restart. Default: 2.
        eta_mult (float): Multiplicative factor of decay in the range of
            learning rates that is applied after each restart. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983


    Scheduler taken from: https://github.com/pytorch/pytorch/pull/11104
    """

    def __init__(self, optimizer, T, eta_min=0, T_mult=2.0, eta_mult=1.0, last_epoch=-1):
        self.T = T
        self.eta_min = eta_min
        self.eta_mult = eta_mult

        if T_mult < 1:
            raise ValueError('T_mult should be >= 1.0.')
        self.T_mult = T_mult

        super(CosineAnnealingRestartsLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs

        if self.T_mult == 1:
            i_restarts = self.last_epoch // self.T
            last_restart = i_restarts * self.T
        else:
            # computation of the last restarting epoch is based on sum of geometric series:
            # last_restart = T * (1 + T_mult + T_mult ** 2 + ... + T_mult ** i_restarts)
            i_restarts = int(math.log(1 - self.last_epoch * (1 - self.T_mult) / self.T,
                                      self.T_mult))
            last_restart = int(self.T * (1 - self.T_mult ** i_restarts) / (1 - self.T_mult))

        if self.last_epoch == last_restart:
            T_i1 = self.T * self.T_mult ** (i_restarts - 1)  # T_{i-1}
            lr_update = self.eta_mult / self._decay(T_i1 - 1, T_i1)
        else:
            T_i = self.T * self.T_mult ** i_restarts
            t = self.last_epoch - last_restart
            lr_update = self._decay(t, T_i) / self._decay(t - 1, T_i)

        return [lr_update * (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    @staticmethod
    def _decay(t, T):
        """Cosine decay for step t in run of length T, where 0 <= t < T."""
        return 0.5 * (1 + math.cos(math.pi * t / T))