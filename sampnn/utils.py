import os
import torch
from tqdm.autonotebook import trange
import shutil
import math
import numpy as np

from torch.optim.optimizer import Optimizer
from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse
from sampnn.data import AverageMeter, Normalizer


def evaluate(generator, model, criterion, optimizer,
             normalizer, device, task="train", verbose=False):
    """
    evaluate the model
    """

    if task == "test":
        model.eval()
        test_targets = []
        test_pred = []
        test_std = []
        test_cif_ids = []
        test_comp = []
    else:
        loss_meter = AverageMeter()
        rmse_meter = AverageMeter()
        mae_meter = AverageMeter()
        if task == "val":
            model.eval()
        elif task == "train":
            model.train()
        else:
            raise NameError("Only train, val or test is allowed as task")

    with trange(len(generator), disable=(not verbose)) as t:
        for input_, target, batch_comp, batch_cif_ids in generator:

            # normalize target
            target_norm = normalizer.norm(target)

            # move tensors to GPU
            input_ = (tensor.to(device) for tensor in input_)
            target_norm = target_norm.to(device)

            # compute output
            output, log_std = model(*input_).chunk(2, dim=1)

            # get predictions and error
            pred = normalizer.denorm(output.data.cpu())

            if task == "test":
                # get the aleatoric std
                std = torch.exp(log_std).data.cpu()*normalizer.std

                # collect the model outputs
                test_cif_ids += batch_cif_ids
                test_comp += batch_comp
                test_targets += target.view(-1).tolist()
                test_pred += pred.view(-1).tolist()
                test_std += std.view(-1).tolist()

            else:
                loss = criterion(output, log_std, target_norm)
                loss_meter.update(loss.data.cpu().item(), target.size(0))

                mae_error = mae(pred, target)
                mae_meter.update(mae_error, target.size(0))

                rmse_error = mse(pred, target).sqrt_()
                rmse_meter.update(rmse_error, target.size(0))

                if task == "train":
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            t.update()

    if task == "test":
        return test_cif_ids, test_comp, test_targets, test_pred, test_std
    else:
        return loss_meter.avg, mae_meter.avg, rmse_meter.avg


def save_checkpoint(state, is_best,
                    checkpoint="checkpoint.pth.tar",
                    best="best.pth.tar"):
    """
    Saves a checkpoint and overwrites the best model when is_best = True
    """

    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)


def load_previous_state(path, model, optimizer=None, 
                        normalizer=None, scheduler=None):
    """
    """
    assert os.path.isfile(path), "no checkpoint found at '{}'".format(path)

    checkpoint = torch.load(path)
    start_epoch = checkpoint["epoch"]
    best_error = checkpoint["best_error"]
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if normalizer:
        normalizer.load_state_dict(checkpoint["normalizer"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    print("Loaded '{}'".format(path, checkpoint["epoch"]))

    return model, optimizer, normalizer, scheduler, best_error, start_epoch


def RobustL1(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = np.sqrt(2.0) * torch.abs(output - target) * \
        torch.exp(- log_std) + log_std
    return torch.mean(loss)


def RobustL2(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = 0.5 * torch.pow(output - target, 2.0) * \
        torch.exp(- 2.0 * log_std) + log_std
    return torch.mean(loss)


class AdamW(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom) )

        return loss


def cyclical_lr(period, max_mul=10, min_mul=0.1, end=-1):
    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: 1. + (max_mul - 1.) * relative(it, period) \
                            if it < end else min_mul

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (period))
        x = abs(2*(it / period - cycle) + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda
