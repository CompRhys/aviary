import os
import gc
import datetime

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split as split
from sklearn.metrics import r2_score

from sampnn.message import CompositionNet, ResidualNetwork, \
                            SimpleNetwork
from sampnn.data import input_parser, CompositionData, \
                        Normalizer, collate_batch
from sampnn.utils import evaluate, save_checkpoint, \
                        load_previous_state, RobustL1, \
                        RobustL2, cyclical_lr, \
                        LRFinder
import math


def init_model(orig_atom_fea_len):

    model = CompositionNet(orig_atom_fea_len,
                           elem_fea_len=args.atom_fea_len,
                           n_graph=args.n_graph)

    model.to(args.device)

    normalizer = Normalizer()

    return model, normalizer


def init_optim(model):

    # Select Loss Function, Note we use Robust loss functions that
    # are used to train an aleatoric error estimate
    if args.loss == "L1":
        criterion = RobustL1
    elif args.loss == "L2":
        criterion = RobustL2
    else:
        raise NameError("Only L1 or L2 are allowed as --loss")

    # Select Optimiser
    if args.optim == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
    elif args.optim == "AdamW":
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    else:
        raise NameError("Only SGD or Adam is allowed as --optim")

    if args.clr_cycles > 0:
        clr = cyclical_lr(period=args.clr_period,
                          cycle_mul=0.1,
                          tune_mul=0.05,
                          end=args.clr_cycles*args.clr_period)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, [clr])
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [])

    return criterion, optimizer, scheduler


def main():

    dataset = CompositionData(data_path=args.data_path,
                              fea_path=args.fea_path)
    orig_atom_fea_len = dataset.atom_fea_dim

    indices = list(range(len(dataset)))
    train_idx, test_idx = split(indices, random_state=args.seed,
                                test_size=args.test_size)

    train_set = torch.utils.data.Subset(dataset, train_idx[0::args.sub_sample])
    test_set = torch.utils.data.Subset(dataset, test_idx)

    if not os.path.isdir("models/"):
        os.makedirs("models/")

    if not os.path.isdir("runs/"):
        os.makedirs("runs/")

    if not os.path.isdir("results/"):
        os.makedirs("results/")

    ensemble(args.fold_id, train_set, test_set,
             args.ensemble, orig_atom_fea_len)


def ensemble(fold_id, dataset, test_set,
             ensemble_folds, fea_len):
    """
    Train multiple models
    """

    params = {"batch_size": args.batch_size,
              "num_workers": args.workers,
              "pin_memory": False,
              "shuffle": True,
              "collate_fn": collate_batch}

    if args.val_size == 0.0:
        print("No validation set used, using test set for evaluation purposes")
        # Note that when using this option care must be taken not to
        # peak at the test-set. The only valid model to use is the one obtained
        # after the final epoch where the epoch count is decided in advance of
        # the experiment.
        train_subset = dataset
        val_subset = test_set
    else:
        indices = list(range(len(dataset)))
        train_idx, val_idx = split(indices, random_state=args.seed,
                                   test_size=args.val_size/(1-args.test_size))
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_generator = DataLoader(train_subset, **params)
    val_generator = DataLoader(val_subset, **params)

    if not args.evaluate:
        if args.lr_search:
            model, normalizer = init_model(fea_len)
            criterion, optimizer, scheduler = init_optim(model)

            if args.fine_tune:
                print("Fine tune from a network trained on a different dataset")
                previous_state = load_previous_state(args.fine_tune,
                                                     model,
                                                     args.device)
                model, _, _, _, _, _ = previous_state
                model.to(args.device)
                criterion, optimizer, scheduler = init_optim(model)
            
            lr_finder = LRFinder(model, optimizer, criterion,
                                 metric="mse", device=args.device)
            lr_finder.range_test(train_generator, end_lr=1,
                                 num_iter=100, step_mode="exp")
            lr_finder.plot()
            return

        for run_id in range(ensemble_folds):

            # this allows us to run ensembles in parallel rather than in series
            # by specifiying the run-id arg.
            if ensemble_folds == 1:
                run_id = args.run_id

            model, normalizer = init_model(fea_len)
            criterion, optimizer, scheduler = init_optim(model)

            _, sample_target, _, _ = collate_batch(train_subset)
            normalizer.fit(sample_target)

            writer = SummaryWriter(log_dir=("runs/f-{f}_r-{r}_s-{s}_t-{t}_"
                                            "{date:%d-%m-%Y_%H:%M:%S}").format(
                                                date=datetime.datetime.now(),
                                                f=fold_id,
                                                r=run_id,
                                                s=args.seed,
                                                t=args.sub_sample))

            experiment(fold_id, run_id, args,
                       train_generator, val_generator,
                       model, optimizer, criterion,
                       normalizer,  scheduler, writer)

    test_ensemble(fold_id, ensemble_folds, test_set, fea_len)


def experiment(fold_id, run_id, args,
               train_generator, val_generator,
               model, optimizer, criterion,
               normalizer,  scheduler, writer):
    """
    for given training and validation sets run an experiment.
    """

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Number of Trainable Parameters: {}".format(num_param))

    checkpoint_file = ("models/checkpoint_"
                       "f-{}_r-{}_s-{}_t-{}.pth.tar").format(fold_id,
                                                             run_id,
                                                             args.seed,
                                                             args.sub_sample)
    best_file = ("models/best_"
                 "f-{}_r-{}_s-{}_t-{}.pth.tar").format(fold_id,
                                                       run_id,
                                                       args.seed,
                                                       args.sub_sample)

    if args.resume:
        print("Resume Training from previous model")
        previous_state = load_previous_state(checkpoint_file,
                                             model,
                                             args.device,
                                             optimizer,
                                             normalizer,
                                             scheduler)
        model, optimizer, normalizer, scheduler, \
            best_mae, start_epoch = previous_state
        model.to(args.device)
    else:
        if args.fine_tune:
            print("Fine tune from a network trained on a different dataset")
            previous_state = load_previous_state(args.fine_tune,
                                                 model,
                                                 args.device)
            model, _, _, _, _, _ = previous_state
            model.to(args.device)
            criterion, optimizer, scheduler = init_optim(model)
        elif args.transfer:
            print("Use model as a feature extractor and retrain last layer")
            previous_state = load_previous_state(args.transfer,
                                                 model,
                                                 args.device)
            model, _, _, _, _, _ = previous_state
            for p in model.parameters():
                p.requires_grad = False
            num_ftrs = model.output_nn.fc_out.in_features
            model.output_nn.fc_out = nn.Linear(num_ftrs, 2)
            model.to(args.device)
            criterion, optimizer, scheduler = init_optim(model)

        _, best_mae, _ = evaluate(generator=val_generator,
                                  model=model,
                                  criterion=criterion,
                                  optimizer=None,
                                  normalizer=normalizer,
                                  device=args.device,
                                  task="val")
        start_epoch = 0

    # try except structure used to allow keyboard interupts to stop training
    # without breaking the code
    try:
        for epoch in range(start_epoch, start_epoch+args.epochs):
            # Training
            t_loss, t_mae, t_rmse = evaluate(generator=train_generator,
                                             model=model,
                                             criterion=criterion,
                                             optimizer=optimizer,
                                             normalizer=normalizer,
                                             device=args.device,
                                             task="train",
                                             verbose=True)

            # Validation
            with torch.no_grad():
                # evaluate on validation set
                val_loss, val_mae, val_rmse = evaluate(generator=val_generator,
                                                       model=model,
                                                       criterion=criterion,
                                                       optimizer=None,
                                                       normalizer=normalizer,
                                                       device=args.device,
                                                       task="val")

            # if epoch % args.print_freq == 0:
            print("Epoch: [{}/{}]\n"
                  "Train      : Loss {:.4f}\t"
                  "MAE {:.3f}\t RMSE {:.3f}\n"
                  "Validation : Loss {:.4f}\t"
                  "MAE {:.3f}\t RMSE {:.3f}\n".format(
                    epoch+1, start_epoch + args.epochs,
                    t_loss, t_mae, t_rmse,
                    val_loss, val_mae, val_rmse))

            is_best = val_mae < best_mae
            if is_best:
                best_mae = val_mae

            checkpoint_dict = {"epoch": epoch,
                               "state_dict": model.state_dict(),
                               "best_error": best_mae,
                               "optimizer": optimizer.state_dict(),
                               "normalizer": normalizer.state_dict(),
                               "scheduler": scheduler.state_dict(),
                               "args": vars(args)}

            save_checkpoint(checkpoint_dict,
                            is_best,
                            checkpoint_file,
                            best_file)

            writer.add_scalar("loss/train", t_loss, epoch+1)
            writer.add_scalar("loss/validation", val_loss, epoch+1)
            writer.add_scalar("rmse/train", t_rmse, epoch+1)
            writer.add_scalar("rmse/validation", val_rmse, epoch+1)
            writer.add_scalar("mae/train", t_mae, epoch+1)
            writer.add_scalar("mae/validation", val_mae, epoch+1)

            scheduler.step()

            # catch memory leak
            gc.collect()

    except KeyboardInterrupt:
        pass

    writer.close()


def test_ensemble(fold_id, ensemble_folds, hold_out_set, fea_len):
    """
    take an ensemble of models and evaluate their performance on the test set
    """

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
          "------------Evaluate model on Test Set------------\n"
          "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    model, normalizer = init_model(fea_len)
    criterion, _, _, = init_optim(model)

    params = {"batch_size": args.batch_size,
              "num_workers": args.workers,
              "pin_memory": False,
              "shuffle": False,
              "collate_fn": collate_batch}

    test_generator = DataLoader(hold_out_set, **params)

    y_ensemble = []
    y_aleatoric = []

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            j = args.run_id
            print("Evaluating Model")
        else:
            print("Evaluating Model {}/{}".format(j+1, ensemble_folds))

        checkpoint = torch.load(f=("models/checkpoint_"
                                   "f-{}_r-{}_s-{}_t-{}"
                                   ".pth.tar").format(fold_id,
                                                      j,
                                                      args.seed,
                                                      args.sub_sample),
                                map_location=args.device)
        model.load_state_dict(checkpoint["state_dict"])
        normalizer.load_state_dict(checkpoint["normalizer"])

        model.eval()
        idx, comp, y_test, pred, std = evaluate(generator=test_generator,
                                                model=model,
                                                criterion=criterion,
                                                optimizer=None,
                                                normalizer=normalizer,
                                                device=args.device,
                                                task="test")

        y_ensemble.append(pred)
        y_aleatoric.append(std)

    y_pred = np.mean(y_ensemble, axis=0)
    y_epistemic = np.var(y_ensemble, axis=0)
    y_aleatoric = np.mean(np.square(y_aleatoric), axis=0)
    y_std = np.sqrt(y_epistemic + y_aleatoric)/np.sqrt(ensemble_folds)

    # calculate metrics and errors with associated errors for ensembles
    res = np.abs(y_test - y_pred)
    mae_avg = np.mean(res)
    mae_std = np.std(res)/np.sqrt(len(res))

    se = np.square(y_test - y_pred)
    mse_avg = np.mean(se)
    mse_std = np.std(se)/np.sqrt(len(se))

    rmse_avg = np.sqrt(mse_avg)
    rmse_std = 0.5 * rmse_avg * mse_std / mse_avg

    print("Ensemble Performance Metrics:")
    print("R2 Score: {:.4f} ".format(r2_score(y_test, y_pred)))
    print("MAE: {:.4f} +/- {:.4f}".format(mae_avg, mae_std))
    print("RMSE: {:.4f} +/- {:.4f}".format(rmse_avg, rmse_std))

    df = pd.DataFrame({"id": idx,
                       "composition": comp,
                       "target": y_test,
                       "mean": y_pred,
                       "std": y_std,
                       "epistemic": np.sqrt(y_epistemic),
                       "aleatoric": np.sqrt(y_aleatoric)})

    if ensemble_folds == 1:
        df.to_csv(index=False,
                  path_or_buf=("results/test_results_"
                               "f-{}_r-{}_s-{}_t-{}"
                               ".pth.tar").format(fold_id,
                                                  args.run_id,
                                                  args.seed,
                                                  args.sub_sample))
    else:
        df.to_csv(index=False,
                  path_or_buf=("results/ensemble_results_"
                               "f-{}_s-{}_t-{}"
                               ".pth.tar").format(fold_id,
                                                  args.seed,
                                                  args.sub_sample))


if __name__ == "__main__":
    args = input_parser()

    print("The model will run on the {} device".format(args.device))

    main()
