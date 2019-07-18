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
            target_var = normalizer.norm(target)
            
            # move tensors to GPU
            input_ = (tensor.to(device) for tensor in input_ )
            target_var = target_var.to(device)

            # compute output
            output = model(*input_)

            loss = criterion(output, target_var)
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



def partitions(number, k):
    """
    Distribution of the folds allowing for cases where 
    the folds do not divide evenly

    Inputs
    --------
    k: int
        The number of folds to split the data into
    number: int
        The number of datapoints in the dataset
    """
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions



def get_indices(n_splits, points):
    """
    Indices of the set test

    Inputs
    --------
    n_splits: int
        The number of folds to split the data into
    points: int
        The number of datapoints in the dataset
    """
    fold_sizes = partitions(points, n_splits)
    indices = np.arange(points).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])



def k_fold_split(n_splits = 3, points = 3001):
    """
    Generates folds for cross validation

    Inputs
    --------
    n_splits: int
        The number of folds to split the data into
    points: int
        The number of datapoints in the dataset

    """
    indices = np.arange(points).astype(int)
    for test_idx in get_indices(n_splits, points):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx



def save_checkpoint(state, is_best, 
                    checkpoint="checkpoint.pth.tar", 
                    best="best.pth.tar" ):
    """
    Saves a checkpoint and overwrites the best model when is_best = True
    """

    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)



# def load_previous_state(path):
#     """
#     """
#     assert os.path.isfile(path), "=> no checkpoint found at '{}'".format(path) 

#     print("Loading Previous Model '{}'".format(path))
#     checkpoint = torch.load(path)
#     args.start_epoch = checkpoint["epoch"]
#     best_error = checkpoint["best_error"]
#     model.load_state_dict(checkpoint["state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer"])
#     normalizer.load_state_dict(checkpoint["normalizer"])
#     print("Loaded Previous Model '{}' (epoch {})"
#             .format(path, checkpoint["epoch"]))

#     return model, optimizer, normalizer, best_error, args.start_epoch