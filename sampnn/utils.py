import os
import torch
from tqdm import trange
import shutil
import numpy as np

from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse
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
        test_preds = []
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
            output, output_err = model(*input_).chunk(2,dim=1)

            # loss = criterion(output, target_norm)
            loss = criterion(output, output_err, target_norm)
            losses.update(loss.data.cpu().item(), target.size(0))

            # measure accuracy and record loss
            pred = normalizer.denorm(output.data.cpu())
            
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
                test_preds += pred.view(-1).tolist()
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
        return test_cif_ids, test_comp, test_targets, test_preds
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


def RobustL1(output, output_std, target):
    loss = np.sqrt(2) * torch.abs(output - target) * \
           torch.exp(-output_std) + output_std
    return torch.mean(loss)

def RobustL2(output, output_var, target):
    loss = torch.pow(output - target, 2) * \
           torch.exp(-output_var) + 0.5 * output_var
    return torch.mean(loss)