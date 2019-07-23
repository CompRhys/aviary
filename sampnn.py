import os
import gc

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

from sampnn.message import CompositionNet, weight_reset
from sampnn.data import input_parser, CompositionData, \
                        Normalizer, collate_batch
from sampnn.utils import evaluate, save_checkpoint, \
                        load_previous_state, RobustL1, \
                        RobustL2


def init_model(orig_atom_fea_len):

    model = CompositionNet(orig_atom_fea_len, 
                            atom_fea_len=args.atom_fea_len,
                            n_graph=args.n_graph)

    model.to(args.device)

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
    else:
        raise NameError("Only SGD or Adam is allowed as --optim")

    normalizer = Normalizer()    

    objects = (model, criterion, optimizer, normalizer)

    return objects


def main():

    if args.debug:
        dataset = CompositionData(data_path=args.data_path, 
                                    fea_path=args.fea_path)
        orig_atom_fea_len = dataset.atom_fea_dim + 1
        
        indices = list(range(len(dataset)))
        train_idx, test_idx = split(indices, test_size=args.test_size, 
                                    train_size=args.train_size,
                                    random_state=0)

        train_set = torch.utils.data.Subset(dataset, train_idx)
        test_set = torch.utils.data.Subset(dataset, test_idx)

    else:
        train_set = CompositionData(data_path="data/datasets/oqmd_train.csv", 
                                    fea_path=args.fea_path)
        test_set = CompositionData(data_path="data/datasets/oqmd_test.csv", 
                                    fea_path=args.fea_path)
        orig_atom_fea_len = train_set.atom_fea_dim + 1

    model_dir = "models/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    ensemble(model_dir, args.fold_id, train_set, test_set, 
                args.ensemble, orig_atom_fea_len)


def ensemble(model_dir, fold_id, dataset, test_set, 
                ensemble_folds, fea_len, test=True):
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
        # Note that when using this option care must be taken not to
        # peak at the test-set. The only valid model to use is the one obtained
        # after the final epoch where the epoch count is decided in advance of
        # the experiment.
        train_subset = dataset
        val_subset = test_set
    else:
        indices = list(range(len(dataset)))
        train_idx, val_idx = split(indices, test_size=args.val_size, 
                                    random_state=0)
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

    train_generator = DataLoader(train_subset, **params)
    val_generator = DataLoader(val_subset, **params)

    if not args.evaluate:
        for run_id in range(ensemble_folds):

            if ensemble_folds == 1:
                run_id = args.run_id

            model, criterion, optimizer, normalizer = init_model(fea_len)

            _, sample_target, _, _ = collate_batch(train_subset)
            normalizer.fit(sample_target)

            experiment(model_dir, fold_id, run_id, args, 
                        train_generator, val_generator, 
                        model, optimizer, criterion, 
                        normalizer)        

    if test:
        test_ensemble(model_dir, fold_id, ensemble_folds, test_set, fea_len)


def experiment(model_dir, fold_id, run_id, args, 
                train_generator, val_generator, 
                model, optimizer, criterion, 
                normalizer):
    """
    for given training and validation sets run an experiment.
    """

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Number of Trainable Parameters: {}".format(num_param))

    writer = SummaryWriter(flush_secs=30)

    checkpoint_file = model_dir+"checkpoint_{}_{}.pth.tar".format(fold_id, run_id)
    best_file = model_dir+"best_{}_{}.pth.tar".format(fold_id, run_id)

    if args.resume:
        previous_state = load_previous_state(checkpoint_file, model, optimizer, normalizer)
        model, optimizer, normalizer, best_error, start_epoch = previous_state
    elif args.transfer:
        previous_state = load_previous_state(args.transfer, model, optimizer, normalizer)
        model, optimizer, normalizer, best_error, start_epoch = previous_state
        model.output_nn.apply(weight_reset)
    else:
        _, best_error = evaluate(generator=val_generator, model=model, 
                        criterion=criterion, optimizer=None, 
                        normalizer=normalizer, device=args.device, 
                        task="val")
        start_epoch = 0

    # try except structure used to allow keyboard interupts to stop training
    # without breaking the code
    try:
        for epoch in range(start_epoch, start_epoch+ args.epochs):
            # Training
            train_loss, train_error = evaluate(generator=train_generator, model=model, 
                                                criterion=criterion, optimizer=optimizer, 
                                                normalizer=normalizer, device=args.device, 
                                                task="train", verbose=True)

            # Validation
            with torch.set_grad_enabled(False):
                # evaluate on validation set
                val_loss, val_error = evaluate(generator=val_generator, model=model, 
                                                criterion=criterion, optimizer=None, 
                                                normalizer=normalizer, device=args.device, 
                                                task="val")

            # if epoch % args.print_freq == 0:
            print("Epoch: [{0}/{1}]\t"
                "Train : Loss {2:.4f}\t"
                "Error {3:.3f}\t"
                "Validation : Loss {4:.4f}\t"
                "Error {5:.3f}\n".format(
                epoch+1, start_epoch + args.epochs, train_loss, train_error,
                val_loss, val_error))

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

            # catch memory leak
            gc.collect()

    except KeyboardInterrupt:
        pass

    writer.close()


def test_ensemble(model_dir, fold_id, ensemble_folds, hold_out_set, fea_len):
    """
    take an ensemble of models and evaluate their performance on the test set
    """

    print(  "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            "------------Evaluate model on Test Set------------\n"
            "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    model, criterion, _, normalizer = init_model(fea_len)

    params = {  "batch_size": args.batch_size,
                "num_workers": args.workers, 
                "pin_memory": False,
                "shuffle":False,
                "collate_fn": collate_batch}
        
    test_generator = DataLoader(hold_out_set, **params)

    y_ensemble = []
    y_aleatoric = []

    for j in range(ensemble_folds):

        print("Model {}/{}".format(j+1, ensemble_folds))

        checkpoint = torch.load(model_dir+"checkpoint_{}_{}.pth.tar".format(fold_id,j))
        model.load_state_dict(checkpoint["state_dict"])
        normalizer.load_state_dict(checkpoint["normalizer"])

        model.eval()
        idx, comp, y_test, pred, std = evaluate(generator=test_generator, model=model, 
                                            criterion=criterion, optimizer=None, 
                                            normalizer=normalizer, device=args.device, 
                                            task="test")

        y_ensemble.append(pred)
        y_aleatoric.append(std)

    y_pred = np.mean(y_ensemble, axis=0)
    y_epistemic = np.var(y_ensemble, axis=0)
    y_aleatoric = np.mean(np.square(y_aleatoric), axis=0)
    y_std = np.sqrt(y_epistemic + y_aleatoric)

    # calculate metrics and errors with associated errors for ensembles
    # errors in the MAE and MSE are estimated using 
    res = np.abs(y_test - y_pred)
    mae_avg = np.mean(res)
    mae_std = np.std(res)/np.sqrt(len(res))

    se = np.square(y_test - y_pred)
    mse_avg = np.mean(se)
    mse_std = np.std(se)/np.sqrt(len(se))

    rmse_avg = np.sqrt(mse_avg)
    rmse_std = 0.5 * rmse_avg * mse_std / mse_avg

    print("Ensemble Performance Metrics:")
    print("R2 Score: {:.4f} ".format(r2_score(y_test,y_pred)))
    print("MAE: {:.4f} +/- {:.4f}".format(mae_avg, mae_std))
    print("RMSE: {:.4f} +/- {:.4f}".format(rmse_avg, rmse_std))

    df = pd.DataFrame({ "id" : idx, 
                        "composition" : comp, 
                        "target" : y_test, 
                        "mean" : y_pred, 
                        "std" : y_std,
                        "epistemic" : np.sqrt(y_epistemic), 
                        "aleatoric" : np.sqrt(y_aleatoric),  
                        })

    df.to_csv("test_results.csv", index=False)


if __name__ == "__main__":
    args = input_parser()
    
    print("The model will run on the {} device".format(args.device))

    main()

