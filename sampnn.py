import argparse
import sys
import os
import warnings
import gc
import copy
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchcontrib.optim import SWA
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from sklearn.model_selection import train_test_split as split

from sampnn.message import CompositionNet
from sampnn.data import input_parser, CompositionData 
from sampnn.data import Normalizer
from sampnn.data import collate_batch
from sampnn.utils import train, evaluate, save_checkpoint
from sampnn.utils import k_fold_split


def init_model(orig_atom_fea_len):

    device = torch.device("cuda") if args.cuda else torch.device("cpu")

    model = CompositionNet(orig_atom_fea_len, 
                            atom_fea_len=args.atom_fea_len,
                            n_graph=args.n_graph)

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Number of Trainable Parameters: {}".format(num_param))

    if args.cuda:
        model.cuda()

    if args.loss == "L1":
        criterion = nn.L1Loss()
    elif args.loss == "Huber":
        criterion = nn.SmoothL1Loss()
    elif args.loss == "L2":
        criterion = nn.MSELoss()
    else:
        raise NameError("Only L1, L2 or Huber is allowed as --loss")
        

    # Choose Optimiser
    if args.optim == "SGD":
        base_optim = optim.SGD(model.parameters(), 
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
    elif args.optim == "Adam":
        base_optim = optim.Adam(model.parameters(), 
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    elif args.optim == "RMSprop":
        base_optim = optim.RMSprop(model.parameters(), 
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    else:
        raise NameError("Only SGD or Adam is allowed as --optim")

    # Note that stochastic weight averaging increases the bias
    # optimizer = SWA(base_optim, swa_start=10, swa_freq=5, swa_lr=0.05)
    optimizer = base_optim

    normalizer = Normalizer()

    objects = (device, model, criterion, optimizer, normalizer)

    return objects



def main():

    # dataset = CompositionData(data_path=args.data_path, fea_path=args.fea_path)
    
    # indices = list(range(len(dataset)))

    # train_idx, test_idx = split(indices, test_size=args.test_size, train_size=args.train_size,
    #                             random_state=0)

    # train_set = torch.utils.data.Subset(dataset, train_idx)
    # test_set = torch.utils.data.Subset(dataset, test_idx)

    train_set = CompositionData(data_path="/data/datasets/oqmd_train.csv", fea_path=args.fea_path)
    test_set = CompositionData(data_path="/data/datasets/oqmd_test.csv", fea_path=args.fea_path)


    orig_atom_fea_len = train_set.atom_fea_dim + 1
    single(5, train_set, test_set, orig_atom_fea_len)

    # ensemble(2, train_set, test_set, 10, orig_atom_fea_len)


def single(fold_id, dataset, test_set, fea_len, test=True):
    """
    Train a single model
    """

    params = {  "batch_size": args.batch_size,
                "num_workers": args.workers, 
                "pin_memory": False,
                "shuffle":False,
                "collate_fn": collate_batch}

    if args.val_size == 0.0:
        print("No validation set used, using test set for evaluation purposes")
        print("{} points for training, {} points for validation".format(len(dataset), len(test_set)))
        train_subset = dataset
        val_subset = test_set
    else:
        print("reserving {}% of test set for evaluation purposes".format(100*args.val_size))
        indices = list(range(len(dataset)))
        train_idx, val_idx = split(indices, test_size=args.val_size, 
                                    random_state=0)
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_generator = DataLoader(dataset, **params)
    val_generator = DataLoader(test_set, **params)

    device, model, criterion, optimizer, normalizer = init_model(fea_len)

    _, sample_target, _ = collate_batch(train_subset)
    normalizer.fit(sample_target)

    experiment(fold_id, 9, args, train_generator, val_generator, 
                model, optimizer, criterion, normalizer)

    if test:
        test_model(fold_id, 9, test_set, fea_len)



def ensemble(fold_id, dataset, test_set, ensemble_folds, fea_len, test=True):
    """
    Train multiple models
    """

    params = {  "batch_size": args.batch_size,
                "num_workers": args.workers, 
                "pin_memory": False,
                "shuffle":False,
                "collate_fn": collate_batch}

    if args.val_size == 0.0:
        print("No validation set used, using test set for evaluation purposes")
        train_subset = dataset
        val_subset = test_set
    else:
        print("reserving {}% of test set for evaluation purposes".format(100*args.val_size))
        indices = list(range(len(dataset)))
        train_idx, val_idx = split(indices, test_size=args.val_size, 
                                    random_state=0)
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_generator = DataLoader(train_subset, **params)
    val_generator = DataLoader(val_subset, **params)

    for run_id in range(ensemble_folds):

        device, model, criterion, optimizer, normalizer = init_model(fea_len)

        _, sample_target, _ = collate_batch(train_subset)
        normalizer.fit(sample_target)

        experiment(fold_id, run_id, args, train_generator, val_generator, 
                    model, optimizer, criterion, normalizer)        

    if test:
        test_ensemble(fold_id, ensemble_folds, test_set, fea_len)


def nested_cv(cv_folds=5):
    """
    Divide the total dataset into X folds.

    Keeping one fold as a hold out set train 
    an ensemble of models on the remaining data.

    Iterate such that each fold is used as the 
    hold out set once and return the cross validation error.
    """

    dataset = CompositionData(args.data_dir, seed=43)

    orig_atom_fea_len = dataset.atom_fea_dim

    total = len(dataset)

    splits = k_fold_split(cv_folds, total)

    for fold_id, (training, hold_out) in enumerate(splits):
        training_set = torch.utils.data.Subset(dataset, training)
        hold_out_set = torch.utils.data.Subset(dataset, hold_out)

        ensemble_folds = 10
        cv_ensemble(fold_id, training_set, ensemble_folds, orig_atom_fea_len)

        break

def cv_ensemble(fold_id, dataset, ensemble_folds, fea_len):
    """
    Divide the dataset into X folds.

    Keeping one fold as a hold-out set train a 
    model on the next fold for a given number of epochs.

    using the hold-out set keep the best 
    performing model over the whole training period.
    """

    params = {  "batch_size": args.batch_size,
                "num_workers": args.workers, 
                "pin_memory": False,
                "shuffle":False,
                "collate_fn": collate_batch}

    total = len(dataset)
    splits = k_fold_split(ensemble_folds, total)

    for run_id, (train, val) in enumerate(splits):

        device, model, criterion, optimizer, normalizer = init_model(fea_len)

        train_subset = torch.utils.data.Subset(dataset, train)
        val_subset = torch.utils.data.Subset(dataset, val)

        train_generator = DataLoader(train_subset, **params)
        val_generator = DataLoader(val_subset, **params)

        _, sample_target, _ = collate_batch(train_subset)
        normalizer.fit(sample_target)

        experiment(fold_id, run_id, args, train_generator, val_generator, 
            model, optimizer, criterion, normalizer)
     

    test_ensemble(fold_id, ensemble_folds, test_set, orig_atom_fea_len)

def experiment(fold_id, run_id, args, train_generator, val_generator, 
                model, optimizer, criterion, normalizer):
    """
    for a given training an validation set run an experiment.

    return the model that performed best on the validation set.
    """

    writer = SummaryWriter()

    _, best_error = evaluate(val_generator, model, criterion, normalizer,
                        verbose=False)

    checkpoint_file = "models/checkpoint_{}_{}.pth.tar".format(fold_id, run_id)
    best_file = "models/best_{}_{}.pth.tar".format(fold_id, run_id)

    try:
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            # Training
            model.train()
            train_loss, train_error = train(train_generator, model, criterion, 
                                            optimizer, normalizer, args.cuda)

            # Validation
            with torch.set_grad_enabled(False):
                # switch to evaluate mode
                model.eval()
                # evaluate on validation set
                val_loss, val_error = evaluate(val_generator, model, criterion, 
                                                normalizer, args.cuda)

            print("Epoch: [{0}/{1}]\t"
                    "Train : Loss {2:.4f}\t"
                    "Error {3:.3f}\t"
                    "Validation : Loss {4:.4f}\t"
                    "Error {5:.3f}\n".format(
                    epoch+1, args.start_epoch + args.epochs, train_loss, train_error,
                    val_loss, val_error))

            # scheduler.step()

            is_best = val_error < best_error
            if is_best:
                best_error = val_error

            checkpoint_dict = { "epoch": epoch + 1,
                                "state_dict": model.state_dict(),
                                "best_error": best_error,
                                "optimizer": optimizer.state_dict(),
                                "normalizer": normalizer.state_dict(),
                                "args": vars(args)
                                }

            save_checkpoint(checkpoint_dict, 
                            is_best,
                            checkpoint_file,
                            best_file)


            writer.add_scalar("data/train", train_error, epoch+1)
            writer.add_scalar("data/validation", val_error, epoch+1)

            if epoch % 25 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch+1)
                    # writer.add_histogram(name+"/grad", param.grad.clone().cpu().data.numpy(), epoch+1)

            # catch memory leak
            gc.collect()

    except KeyboardInterrupt:
        pass

    

def test_model(fold_id, run_id, hold_out_set, fea_len):
    """
    """

    print(  "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "------------Evaluate Model on Test Set------------\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    device, model, criterion, _, normalizer = init_model(fea_len)

    params = {  "batch_size": args.batch_size,
                "num_workers": args.workers, 
                "pin_memory": False,
                "shuffle":False,
                "collate_fn": collate_batch}
        
    test_generator = DataLoader(hold_out_set, **params)


    # best_checkpoint = torch.load("models/best_{}_{}.pth.tar".format(fold_id,run_id))
    best_checkpoint = torch.load("models/checkpoint_{}_{}.pth.tar".format(fold_id,run_id))
    model.load_state_dict(best_checkpoint["state_dict"])
    normalizer.load_state_dict(best_checkpoint["normalizer"])

    # print("The best model performance on the validation" 
    #     " set occured on epoch {}".format(best_checkpoint["epoch"]))

    model.eval()
    ids, targets, preds = evaluate(test_generator, model, criterion, normalizer, test=True)

    with open("test_results.csv", "w") as f:
        writer = csv.writer(f)
        for id_, target, pred in zip(ids, targets, preds):
            writer.writerow((id_, target, pred))



def test_ensemble(fold_id, ensemble_folds, hold_out_set, fea_len):
    """
    """

    print(  "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "----------Evaluate Ensemble on Test Set-----------\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    device, model, criterion, _, normalizer = init_model(fea_len)

    params = {  "batch_size": args.batch_size,
                "num_workers": args.workers, 
                "pin_memory": False,
                "shuffle":False,
                "collate_fn": collate_batch}
        
    test_generator = DataLoader(hold_out_set, **params)

    ensemble_preds = []

    for j in range(ensemble_folds):

        # best_checkpoint = torch.load("models/best_{}_{}.pth.tar".format(fold_id,j))
        best_checkpoint = torch.load("models/checkpoint_{}_{}.pth.tar".format(fold_id,j))
        model.load_state_dict(best_checkpoint["state_dict"])
        normalizer.load_state_dict(best_checkpoint["normalizer"])

        # print(best_checkpoint["normalizer"])
        # print("The best model performance on the validation" 
        #     " set occured on epoch {}".format(best_checkpoint["epoch"]))

        model.eval()
        id_, tar_, pred_ = evaluate(test_generator, model, criterion, normalizer, test=True)

        ensemble_preds.append(pred_)

    ensemble_preds = np.array(ensemble_preds)
    mean_preds = np.mean(ensemble_preds, axis=0).tolist()
    mean_std = np.std(ensemble_preds, axis=0).tolist()

    with open("test_results.csv", "w") as f:
        writer = csv.writer(f)
        for id_, target, pred, std in zip(id_, tar_, mean_preds, mean_std):
            writer.writerow((id_, target, pred, std))




if __name__ == "__main__":
    args = input_parser()

    # nested_cv()
    main()

