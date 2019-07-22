import os
import torch
from tqdm.autonotebook import trange
import shutil
import math
import numpy as np

from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse
from torch.optim.lr_scheduler import _LRScheduler
from sampnn.data import AverageMeter, Normalizer

def evaluate(generator, model, criterion, optimizer, 
            normalizer, device, task="train", verbose=False):
    """ 
    evaluate the model 
    """

    losses = AverageMeter()
    errors = AverageMeter()

    if task == "train":
        model.train()
        leave = False
    elif task == "val":
        model.eval()
    elif task == "test":
        model.eval()
        test_targets = []
        test_pred = []
        test_std = []
        test_cif_ids = []
        test_comp = []
    else:
        raise NameError("Only train, val or test is allowed as task")
    
    with trange(len(generator), disable=(not verbose), position=1) as t:
        for input_, target, batch_comp, batch_cif_ids in generator:
            
            # normalize target
            target_norm = normalizer.norm(target)
            
            # move tensors to GPU
            input_ = (tensor.to(device) for tensor in input_ )
            target_norm = target_norm.to(device)

            # compute output
            # output = model(*input_)
            output, log_std = model(*input_).chunk(2,dim=1)

            # loss = criterion(output, target_norm)
            loss = criterion(output, log_std, target_norm)
            losses.update(loss.data.cpu().item(), target.size(0))

            # measure accuracy and record loss
            pred = normalizer.denorm(output.data.cpu())
            std = torch.exp(log_std).data.cpu()*normalizer.std
            
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
                test_std += std.view(-1).tolist()
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
        return test_cif_ids, test_comp, test_targets, test_pred, test_std
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
    loss = torch.pow(output - target, 2.0) * \
           torch.exp(- 2.0 * log_std) + log_std
    return torch.mean(loss)
