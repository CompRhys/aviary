import os
import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split as split
from sklearn.metrics import r2_score

from roost.model import Roost
from roost.data import input_parser, CompositionData, \
                        collate_batch
from roost.utils import load_previous_state, Normalizer,\
                        RobustL1, RobustL2


def init_model(dataset):

    model = Roost(elem_emb_len=dataset.atom_emb_len,
                    elem_fea_len=args.atom_fea_len,
                    n_graph=args.n_graph,
                    device=args.device)

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Number of Trainable Parameters: {}".format(num_param))

    # TODO parallelise the code over multiple GPUs
    # if (torch.cuda.device_count() > 1) and (args.device==torch.device("cuda")):
    #     print("The model will use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

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

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [])

    return criterion, optimizer, scheduler


def main():

    dataset = CompositionData(data_path=args.data_path,
                              fea_path=args.fea_path)

    train_idx = list(range(len(dataset)))

    if args.test_path:
        print(f"using independent test set: {args.test_path}")
        test_set = CompositionData(data_path=args.test_path,
                                    fea_path=args.fea_path)
        test_set = torch.utils.data.Subset(test_set, range(len(test_set)))
    else:
        print(f"using {args.test_size} of training set as test set")
        train_idx, test_idx = split(train_idx, random_state=args.seed,
                                    test_size=args.test_size)
        test_set = torch.utils.data.Subset(dataset, test_idx)

    if args.val_path:
        print(f"using independent validation set: {args.val_path}")
        val_set = CompositionData(data_path=args.val_path,
                                    fea_path=args.fea_path)
        val_set = torch.utils.data.Subset(val_set, range(len(val_set)))
    else:
        if args.val_size == 0.0:
            print("No validation set used, using test set for evaluation purposes")
            # Note that when using this option care must be taken not to
            # peak at the test-set. The only valid model to use is the one obtained
            # after the final epoch where the epoch count is decided in advance of
            # the experiment.
            val_set = test_set
        else:
            print(f"using {args.val_size} of training set as validation set")
            train_idx, val_idx = split(train_idx, random_state=args.seed,
                                    test_size=args.val_size/(1-args.test_size))
            val_set = torch.utils.data.Subset(dataset, val_idx)

    train_set = torch.utils.data.Subset(dataset, train_idx[0::args.sample])

    if not os.path.isdir("models/"):
        os.makedirs("models/")

    if not os.path.isdir("runs/"):
        os.makedirs("runs/")

    if not os.path.isdir("results/"):
        os.makedirs("results/")

    if not args.evaluate:
        train_ensemble(args.data_id, args.ensemble, train_set, val_set)

    test_ensemble(args.data_id, args.ensemble, test_set)


def train_ensemble(data_id, ensemble_folds, train_set, val_set):
    """
    Train multiple models
    """

    params = {"batch_size": args.batch_size,
              "num_workers": args.workers,
              "pin_memory": False,
              "shuffle": True,
              "collate_fn": collate_batch}

    train_generator = DataLoader(train_set, **params)
    val_generator = DataLoader(val_set, **params)

    for run_id in range(ensemble_folds):
        # this allows us to run ensembles in parallel rather than in series
        # by specifiying the run-id arg.
        if ensemble_folds == 1:
            run_id = args.run_id

        model, normalizer = init_model(train_set.dataset)
        criterion, optimizer, scheduler = init_optim(model)

        sample_target = torch.Tensor(train_set.dataset.df.iloc[:, 2].values)
        normalizer.fit(sample_target)

        writer = SummaryWriter(log_dir=("runs/{f}_r-{r}_s-{s}_t-{t}_"
                                        "{date:%d-%m-%Y_%H:%M:%S}").format(
                                            date=datetime.datetime.now(),
                                            f=data_id,
                                            r=run_id,
                                            s=args.seed,
                                            t=args.sample))

        experiment(data_id, run_id, train_generator, val_generator,
                    model, optimizer, criterion, normalizer,  scheduler, writer)


def experiment(data_id, run_id,
               train_generator, val_generator,
               model, optimizer, criterion,
               normalizer, scheduler, writer):
    """
    for given training and validation sets run an experiment.
    """
    checkpoint_file = (f"models/checkpoint_{data_id}_r-{run_id}"
                       f"_s-{args.seed}_t-{args.sample}.pth.tar")

    best_file = (f"models/best_{data_id}_r-{run_id}"
                 f"_s-{args.seed}_t-{args.sample}.pth.tar")

    if args.resume:
        print(f"Resume Training from {checkpoint_file}")
        previous_state = load_previous_state(checkpoint_file,
                                             model,
                                             args.device,
                                             optimizer,
                                             normalizer,
                                             scheduler)
        model, optimizer, normalizer, scheduler, start_epoch, best_mae = previous_state
        model.epoch = start_epoch
        model.best_mae = best_mae
        model.to(args.device)
    else:
        if args.fine_tune:
            print(f"Fine tune from {args.fine_tune}")
            previous_state = load_previous_state(args.fine_tune,
                                                 model,
                                                 args.device)
            model, _, _, _, _, _ = previous_state
            model.to(args.device)
            criterion, optimizer, scheduler = init_optim(model)
        elif args.transfer:
            print(f"Use {args.transfer} as a feature extractor and retrain last layer")
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

        _, best_mae, _ = model.evaluate(generator=val_generator,
                                  criterion=criterion,
                                  optimizer=None,
                                  normalizer=normalizer,
                                  task="val")
        model.epoch = 1
        model.best_mae = best_mae

    model.fit(train_generator, val_generator, optimizer,
            scheduler, args.epochs, criterion, normalizer,
            writer, checkpoint_file, best_file)


def test_ensemble(data_id, ensemble_folds, hold_out_set):
    """
    take an ensemble of models and evaluate their performance on the test set
    """

    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
          "------------Evaluate model on Test Set------------\n"
          "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    params = {"batch_size": args.batch_size,
              "num_workers": args.workers,
              "pin_memory": False,
              "shuffle": False,
              "collate_fn": collate_batch}

    test_generator = DataLoader(hold_out_set, **params)

    model, normalizer = init_model(hold_out_set.dataset)
    criterion, _, _, = init_optim(model)

    y_ensemble = np.zeros((ensemble_folds, len(hold_out_set)))
    y_aleatoric = np.zeros((ensemble_folds, len(hold_out_set)))

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            j = args.run_id
            print("Evaluating Model")
        else:
            print("Evaluating Model {}/{}".format(j+1, ensemble_folds))

        checkpoint = torch.load(f=(f"models/checkpoint_{data_id}_r-{j}_"
                                   f"s-{args.seed}_t-{args.sample}.pth.tar"),
                                map_location=args.device)

        model.load_state_dict(checkpoint["state_dict"])
        normalizer.load_state_dict(checkpoint["normalizer"])

        model.eval()
        idx, comp, y_test, pred, std = model.evaluate(generator=test_generator,
                                                    criterion=criterion,
                                                    optimizer=None,
                                                    normalizer=normalizer,
                                                    task="test")

        if ensemble_folds == 1:
            j = 0

        y_ensemble[j,:] = pred
        y_aleatoric[j,:] = std

    res = y_ensemble - y_test
    mae = np.mean(np.abs(res), axis=1)
    mse = np.mean(np.square(res), axis=1)
    rmse = np.sqrt(mse)
    r2 = 1 - mse/np.var(y_ensemble)

    if ensemble_folds == 1:
        print("\nModel Performance Metrics:")
        print("R2 Score: {:.4f} ".format(r2[0]))
        print("MAE: {:.4f}".format(mae[0]))
        print("RMSE: {:.4f}".format(rmse[0]))
    else:
        r2_avg = np.mean(r2)
        r2_std = np.std(r2)

        mae_avg = np.mean(mae)
        mae_std = np.std(mae)

        rmse_avg = np.mean(rmse)
        rmse_std = np.std(rmse)

        print("\nModel Performance Metrics:")
        print("R2 Score: {:.4f} +/- {:.4f}".format(r2_avg, r2_std))
        print("MAE: {:.4f} +/- {:.4f}".format(mae_avg, mae_std))
        print("RMSE: {:.4f} +/- {:.4f}".format(rmse_avg, rmse_std))

        # calculate metrics and errors with associated errors for ensembles
        y_ens = np.mean(y_ensemble, axis=0)

        ae = np.abs(y_test - y_ens)
        mae_ens = np.mean(ae)

        se = np.square(y_test - y_ens)
        rmse_ens = np.sqrt(np.mean(se))

        print("\nEnsemble Performance Metrics:")
        print("R2 Score: {:.4f} ".format(r2_score(y_test, y_ens)))
        print("MAE: {:.4f}".format(mae_ens))
        print("RMSE: {:.4f}".format(rmse_ens))

    core = {"id": idx, "composition": comp, "target": y_test}
    results = {"pred-{}".format(num): values for (num, values)
                in enumerate(y_ensemble)}
    errors = {"aleatoric-{}".format(num): values for (num, values)
                in enumerate(y_aleatoric)}

    df = pd.DataFrame({**core, **results, **errors})

    if ensemble_folds == 1:
        df.to_csv(index=False,
                  path_or_buf=(f"results/test_results_{data_id}_"
                               f"r-{args.run_id}_s-{args.seed}_"
                               f"t-{args.sample}.csv"))
    else:
        df.to_csv(index=False,
                  path_or_buf=(f"results/ensemble_results_{data_id}_"
                               f"s-{args.seed}_t-{args.sample}.csv"))


if __name__ == "__main__":
    args = input_parser()

    print(f"The model will run on the {args.device} device")

    main()
