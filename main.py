import argparse
import sys
import os
import csv
import shutil
import time
import warnings
from random import sample

import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from sampnn.message import CompositionNet
from sampnn.data import input_parser, CompositionData 
from sampnn.data import collate_batch, get_data_loaders

args = input_parser()

def main():
    global args, best_mae_error

    # load data
    dataset = CompositionData(*args.data_options)
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset=dataset, batch_size=args.batch_size,
        train_size=args.train_size, num_workers=args.workers,
        val_size=args.val_size, test_size=args.test_size,
        pin_memory=args.cuda)


    # for large data sets we can use a subset for the normaliser
    _, sample_target, _ = collate_batch(dataset)
    normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CompositionNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args.atom_fea_len,
                                n_graph=args.n_conv,
                                h_fea_list=args.h_fea_list)
    if args.cuda:
        model.cuda()

    # Loss Function
    criterion = nn.MSELoss()

    # Choose Optimiser
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # decay the learning rate multiplicatively by gamma every time a 
    # milestone is reached.
    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)

    best_mae_error = validate(val_loader, model, criterion, normalizer,
                                verbose=False)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        losses, mae_errors = train(train_loader, model, criterion, optimizer, epoch, normalizer)

        print('Epoch: [{0}/{1}]\t'
                'Loss {loss.avg:.4f}\t'
                'MAE {mae_errors.avg:.3f}'.format(
                epoch, args.epochs, loss=losses, mae_errors=mae_errors))

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mae_error': best_mae_error,
                'optimizer': optimizer.state_dict(),
                'normalizer': normalizer.state_dict(),
                'args': vars(args)
            }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, test=True)


def train(train_loader, model, criterion, optimizer, 
            epoch, normalizer, verbose = False):
    """
    run a forward pass, backwards pass and then update weights
    """
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target, _) in enumerate(train_loader):
        if args.cuda:
            input_var = (Variable(input[0].cuda(async=True)),
                         Variable(input[1].cuda(async=True)),
                         input[2].cuda(async=True),
                         input[3].cuda(async=True),
                         [atom_idx.cuda(async=True) for atom_idx in input[4]],
                         [crys_idx.cuda(async=True) for crys_idx in input[5]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3],
                         input[4],
                         input[5])

        # normalize target
        target_normed = normalizer.norm(target)

        if args.cuda:
            target_var = Variable(target_normed.cuda(async=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose:
            print('Batch: [{0}/{1}]\t'
                    'Loss {loss.val:.4f}\t'
                    'MAE {mae_errors.val:.3f}'.format(
                    i, len(train_loader), loss=losses, mae_errors=mae_errors))

    return losses, mae_errors
    

def validate(val_loader, model, criterion, normalizer, test=False, verbose=True):

    losses = AverageMeter()
    mae_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            input_var = (Variable(input[0].cuda(async=True)),
                         Variable(input[1].cuda(async=True)),
                         input[2].cuda(async=True),
                         input[3].cuda(async=True),
                         [atom_idx.cuda(async=True) for atom_idx in input[4]],
                         [crys_idx.cuda(async=True) for crys_idx in input[5]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3],
                         input[4],
                         input[5])

        target_normed = normalizer.norm(target)

        if args.cuda:
            target_var = Variable(target_normed.cuda(async=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))
        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

    if test:
       label = 'Test'
    else:
        label = 'Validate'

    if verbose:
        print('{0}: \t'
            'Loss {loss.avg:.4f}\t'
            'MAE {mae_errors.avg:.3f}\n'.format(
            label, loss=losses, mae_errors=mae_errors))

    if test:  
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
                                                 
    return mae_errors.avg


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    '''
    Saves a checkpoint and overwrites the best model when is_best = True
    '''
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Normalizer(object):
    """Normalize a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

if __name__ == '__main__':
    main()
