import os
import datetime

import numpy as np
import pandas as pd

import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, \
                            precision_recall_fscore_support
from sklearn.model_selection import train_test_split as split

from roost.model import Roost, ResidualNetwork
from roost.data import input_parser, CompositionData, collate_batch
from roost.utils import load_previous_state, Normalizer, RobustL1, RobustL2


def main(
    data_path,
    fea_path,
    task,
    loss,
    robust,
    model_name="roost",
    elem_fea_len=64,
    n_graph=3,
    ensemble=1,
    run_id=1,
    seed=42,
    epochs=100,
    log=True,
    sample=1,
    test_size=0.2,
    test_path=None,
    val_size=0.0,
    val_path=None,
    resume=None,
    fine_tune=None,
    transfer=None,
    train=True,
    evaluate=True,
    optim="AdamW",
    learning_rate=3e-4,
    momentum=0.9,
    weight_decay=1e-6,
    batch_size=128,
    workers=0,
    device=torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu"),
    **kwargs,
):

    dataset = CompositionData(data_path=data_path, fea_path=fea_path,
                                task=task)
    n_targets = dataset.n_targets
    elem_emb_len = dataset.elem_emb_len

    train_idx = list(range(len(dataset)))

    if evaluate:
        if test_path:
            print(f"using independent test set: {test_path}")
            test_set = CompositionData(data_path=test_path, fea_path=fea_path)
            test_set = torch.utils.data.Subset(test_set, range(len(test_set)))
        elif test_size == 0.0:
            raise ValueError("test-size must be non-zero to evaluate model")
        else:
            print(f"using {test_size} of training set as test set")
            train_idx, test_idx = split(
                train_idx, random_state=seed, test_size=test_size
            )
            test_set = torch.utils.data.Subset(dataset, test_idx)

    if train:
        if val_path:
            print(f"using independent validation set: {val_path}")
            val_set = CompositionData(data_path=val_path, fea_path=fea_path)
            val_set = torch.utils.data.Subset(val_set, range(len(val_set)))
        else:
            if val_size == 0.0 and evaluate:
                print("No validation set used, using test set for evaluation purposes")
                # NOTE: that when using this option care must be taken not to
                # peak at the test-set. The only valid model to use is the one obtained
                # after the final epoch where the epoch count is decided in advance of
                # the experiment.
                val_set = test_set
            elif val_size == 0.0:
                val_set = None
            else:
                print(f"using {val_size} of training set as validation set")
                train_idx, val_idx = split(
                    train_idx,
                    random_state=seed,
                    test_size=val_size / (1 - test_size),
                )
                val_set = torch.utils.data.Subset(dataset, val_idx)

        train_set = torch.utils.data.Subset(dataset, train_idx[0::sample])

    # print(len(train_set), len(val_set), len(test_set))
    # exit()

    data_params = {
        "batch_size": batch_size, # NOTE could speed up inference by
                                  # having larger batches for val/test
        "num_workers": workers,
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": collate_batch,
    }

    model_params = {
        "task": task,
        "robust": robust,
        "elem_emb_len": elem_emb_len,
        "elem_fea_len": elem_fea_len,
        "n_graph": n_graph,
        "n_targets": n_targets,
        "loss": loss,
        "optim": optim,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "device": device,
        "resume": resume,
        "fine_tune": fine_tune,
        "transfer": transfer,
    }

    if not os.path.isdir("models/"):
        os.makedirs("models/")
    if not os.path.isdir(f"models/{model_name}/"):
        os.makedirs(f"models/{model_name}/")

    if log:
        if not os.path.isdir("runs/"):
            os.makedirs("runs/")

    if not os.path.isdir("results/"):
        os.makedirs("results/")

    if train:
        train_ensemble(
            model_name=model_name,
            run_id=run_id,
            ensemble_folds=ensemble,
            epochs=epochs,
            train_set=train_set,
            val_set=val_set,
            log=log,
            data_params=data_params,
            model_params=model_params
        )

    if evaluate:

        reset = {
            "resume": None,
            "fine_tune": None,
            "transfer": None,
        }
        model_params.update(reset)

        if task == "regression":
            results_regression(
                model_name=model_name,
                run_id=run_id,
                ensemble_folds=ensemble,
                test_set=test_set,
                data_params=data_params,
                model_params=model_params,
                eval_type="checkpoint"
                )
        elif task == "classification":
            results_classification(
                model_name=model_name,
                run_id=run_id,
                ensemble_folds=ensemble,
                test_set=test_set,
                data_params=data_params,
                model_params=model_params,
                eval_type="checkpoint"
                )


def init_model(
    model_name,
    run_id,
    task,
    robust,
    elem_emb_len,
    elem_fea_len,
    n_graph,
    n_targets,
    loss,
    optim,
    learning_rate,
    weight_decay,
    momentum,
    device,
    resume=None,
    fine_tune=None,
    transfer=None,
):

    model = Roost(
        elem_emb_len=elem_emb_len,
        elem_fea_len=elem_fea_len,
        n_graph=n_graph,
        n_targets=n_targets,
        task=task,
        robust=robust,
        device=device,
    )

    model.to(device)

    if fine_tune is not None:
        print(f"Using {fine_tune} as a starting point for fine-tuning")
        previous_state = load_previous_state(fine_tune, model, device)
        model, _, _, _, _, _ = previous_state
        model.epoch = 1
        model.best_val_score = None

    if transfer is not None:
        # TODO: currently if you use a model as a feature extractor and then
        # resume for a checkpoint of that model the material_nn unfreezes.
        print(f"Using {transfer} as a feature extractor and retrain the output_nn")
        previous_state = load_previous_state(transfer, model, device)
        model, _, _, _, _, _ = previous_state
        for p in model.material_nn.parameters():
            p.requires_grad = False

        model.output_nn = ResidualNetwork(
            input_dim=elem_fea_len,
            hidden_layer_dims=[1024, 512, 256, 128, 64],
            output_dim=n_targets,
        )

        model.epoch = 1
        model.best_val_score = None

    # Select Optimiser
    if optim == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optim == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optim == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise NameError("Only SGD, Adam or AdamW are allowed as --optim")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [])

        # Select Loss Function
    if task == "classification":
        normalizer = None

        if robust:
            raise NotImplementedError(
        "Robust/Hetroskedastic classifaction is not currently implemented"
        )
        else:
            criterion = CrossEntropyLoss()

    elif task == "regression":
        normalizer = Normalizer()

        # NOTE: when using Robust loss functions we also get an
        # aleatoric error estimate hence n_targets * 2
        if robust:
            if loss == "L1":
                criterion = RobustL1
            elif loss == "L2":
                criterion = RobustL2
            else:
                raise NameError(
                "Only L1 or L2 losses are allowed for robust regression tasks"
                )
        else:
            if loss == "L1":
                criterion = L1Loss()
            elif loss == "L2":
                criterion = MSELoss()
            else:
                raise NameError(
                "Only L1 or L2 losses are allowed for regression tasks"
                )

    if args.resume:
        checkpoint_file = f"models/{model_name}/checkpoint-r{run_id}.pth.tar"
        print(f"Resuming training from {checkpoint_file}")
        previous_state = load_previous_state(
            checkpoint_file, model, device, optimizer, normalizer, scheduler
        )
        model, optimizer, normalizer = previous_state[:3]
        scheduler, start_epoch, best_val_score = previous_state[3:]
        model.epoch = start_epoch
        model.best_val_score = best_val_score

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Number of Trainable Parameters: {}".format(num_param))

    # TODO: parallelise the code over multiple GPUs. Currently DataParallel
    # crashes as subsets of the batch have different sizes due to the use of
    # lists of lists rather the zero-padding.
    # if (torch.cuda.device_count() > 1) and (device==torch.device("cuda")):
    #     print("The model will use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model.to(device)

    return model, criterion, optimizer, scheduler, normalizer


def train_ensemble(
    model_name,
    run_id,
    ensemble_folds,
    epochs,
    train_set,
    val_set,
    log,
    data_params,
    model_params
):
    """
    Train multiple models
    """

    train_generator = DataLoader(train_set, **data_params)

    if val_set is not None:
        val_generator = DataLoader(val_set, **data_params)
    else:
        val_generator = None

    for run in range(ensemble_folds):
        #  this allows us to run ensembles in parallel rather than in series
        #  by specifiying the run-id arg.
        if ensemble_folds == 1:
            run = run_id

        model, criterion, optimizer, scheduler, normalizer = init_model(
            model_name=model_name, run_id=run, **model_params
        )

        if model.task == "regression":
            sample_target = torch.Tensor(
                train_set.dataset.df.iloc[train_set.indices, 2].values
            )
            normalizer.fit(sample_target)

        if log:
            writer = SummaryWriter(
                log_dir=(
                    f"runs/{model_name}-r{run}_""{date:%d-%m-%Y_%H-%M-%S}"
                ).format(date=datetime.datetime.now())
            )
        else:
            writer = None

        if val_set is not None and model.best_val_score is None:
            with torch.no_grad():
                _, v_metrics = model.evaluate(
                    generator=val_generator,
                    criterion=criterion,
                    optimizer=None,
                    normalizer=normalizer,
                    action="val",
                )
                model.best_val_score = v_metrics[model.scoring_rule]

        model.fit(
            train_generator=train_generator,
            val_generator=val_generator,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            criterion=criterion,
            normalizer=normalizer,
            model_name=model_name,
            run_id=run_id,
            writer=writer,
        )


def results_regression(
    model_name,
    run_id,
    ensemble_folds,
    test_set,
    data_params,
    model_params,
    eval_type="checkpoint",
):
    """
    take an ensemble of models and evaluate their performance on the test set
    """

    print(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        "------------Evaluate model on Test Set------------\n"
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    )

    test_generator = DataLoader(test_set, **data_params)

    model, _, _, _, normalizer = init_model(
        model_name=model_name, run_id=run_id, **model_params
    )

    y_ensemble = np.zeros((ensemble_folds, len(test_set)))
    y_aleatoric = np.zeros((ensemble_folds, len(test_set)))

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            j = run_id
            print("Evaluating Model")
        else:
            print("Evaluating Model {}/{}".format(j + 1, ensemble_folds))

        checkpoint = torch.load(
            f=(f"models/{model_name}/{eval_type}-r{j}.pth.tar"),
            map_location=model_params["device"],
        )

        if ensemble_folds == 1:
            j = 0

        model.load_state_dict(checkpoint["state_dict"])
        normalizer.load_state_dict(checkpoint["normalizer"])

        with torch.no_grad():
            idx, comp, y_test, output = model.predict(
                generator=test_generator,
            )

        if model.robust:
            pred, log_std = output.chunk(2, dim=1)
            pred = normalizer.denorm(pred.data.cpu())
            std = torch.exp(log_std).data.cpu()*normalizer.std
            y_aleatoric[j, :] = std.view(-1).numpy()
        else:
            pred = normalizer.denorm(output.data.cpu())

        y_ensemble[j, :] = pred.view(-1).numpy()

    res = y_ensemble - y_test
    mae = np.mean(np.abs(res), axis=1)
    mse = np.mean(np.square(res), axis=1)
    rmse = np.sqrt(mse)
    r2 = 1 - mse / np.var(y_ensemble, axis=1)

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
        print(f"R2 Score: {r2_avg:.4f} +/- {r2_std:.4f}")
        print(f"MAE: {mae_avg:.4f} +/- {mae_std:.4f}")
        print(f"RMSE: {rmse_avg:.4f} +/- {rmse_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        y_ens = np.mean(y_ensemble, axis=0)

        mae_ens = np.abs(y_test - y_ens).mean()
        mse_ens = np.square(y_test - y_ens).mean()
        rmse_ens = np.sqrt(mse)

        r2_ens = 1 - mse_ens / np.var(y_ens)

        print("\nEnsemble Performance Metrics:")
        print("R2 Score: {:.4f} ".format(r2_score(y_test, y_ens)))
        print(f"R2 Score: {r2_ens:.4f} ")
        print(f"MAE: {mae_ens:.4f}")
        print(f"RMSE: {rmse_ens:.4f}")

    core = {"id": idx, "composition": comp, "target": y_test}
    results = {f"pred-{num}": val for (num, val) in enumerate(y_ensemble)}
    if model.robust:
        results.update({f"aleatoric-{num}": val for (num, val) in enumerate(y_aleatoric)})

    df = pd.DataFrame({**core, **results})

    if ensemble_folds == 1:
        df.to_csv(
            index=False,
            path_or_buf=(f"results/test_results_{model_name}_r-{run_id}.csv"),
        )
    else:
        df.to_csv(
            index=False, path_or_buf=(f"results/ensemble_results_{model_name}.csv")
        )


def results_classification(
    model_name,
    run_id,
    ensemble_folds,
    test_set,
    data_params,
    model_params,
    eval_type="checkpoint",
):
    """
    take an ensemble of models and evaluate their performance on the test set
    """

    print(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        "------------Evaluate model on Test Set------------\n"
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    )

    test_generator = DataLoader(test_set, **data_params)

    model, _, _, _, _ = init_model(
        model_name=model_name, run_id=run_id, **model_params
    )

    y_ensemble = np.zeros((ensemble_folds, len(test_set), model.n_targets))

    metrics = np.zeros((ensemble_folds, 5))

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            j = run_id
            print("Evaluating Model")
        else:
            print("Evaluating Model {}/{}".format(j + 1, ensemble_folds))

        checkpoint = torch.load(
            f=(f"models/{model_name}/{eval_type}-r{j}.pth.tar"),
            map_location=model_params["device"],
        )

        if ensemble_folds == 1:
            j = 0

        model.load_state_dict(checkpoint["state_dict"])

        with torch.no_grad():
            idx, comp, y_test, output = model.predict(
                generator=test_generator,
            )
        
        if model.robust:
            raise NotImplementedError(
        "Robust/Hetroskedastic classifaction is not currently implemented"
        )
        else:
            pred = output.data.cpu()
            logits = softmax(pred, dim=1)

        y_ensemble[j, :, :] = logits.numpy()

        y_test_ohe = np.zeros_like(pred)
        y_test_ohe[np.arange(y_test.size), y_test] = 1

        print(accuracy_score(y_test, np.argmax(logits, axis=1)))
        print(roc_auc_score(y_test_ohe, logits))
        print(precision_recall_fscore_support(y_test, np.argmax(pred, axis=1))[:3])

        # metrics[j,0] = accuracy_score(y_test, np.argmax(logits, axis=1))
        # metrics[j,1] = roc_auc_score(y_test_ohe, logits)
        # metrics[j,2:] = precision_recall_fscore_support(y_test, np.argmax(pred, axis=1))[:3]
        
    print(metrics)

    # res = y_ensemble - y_test
    # mae = np.mean(np.abs(res), axis=1)
    # mse = np.mean(np.square(res), axis=1)
    # rmse = np.sqrt(mse)
    # r2 = 1 - mse / np.var(y_ensemble, axis=1)

    # if ensemble_folds == 1:
    #     print("\nModel Performance Metrics:")
    #     print("R2 Score: {:.4f} ".format(r2[0]))
    #     print("MAE: {:.4f}".format(mae[0]))
    #     print("RMSE: {:.4f}".format(rmse[0]))
    # else:
    #     r2_avg = np.mean(r2)
    #     r2_std = np.std(r2)

    #     mae_avg = np.mean(mae)
    #     mae_std = np.std(mae)

    #     rmse_avg = np.mean(rmse)
    #     rmse_std = np.std(rmse)

    #     print("\nModel Performance Metrics:")
    #     print(f"R2 Score: {r2_avg:.4f} +/- {r2_std:.4f}")
    #     print(f"MAE: {mae_avg:.4f} +/- {mae_std:.4f}")
    #     print(f"RMSE: {rmse_avg:.4f} +/- {rmse_std:.4f}")

    #     # calculate metrics and errors with associated errors for ensembles
    #     y_ens = np.mean(y_ensemble, axis=0)

    #     mae_ens = np.abs(y_test - y_ens).mean()
    #     mse_ens = np.square(y_test - y_ens).mean()
    #     rmse_ens = np.sqrt(mse)

    #     r2_ens = 1 - mse_ens / np.var(y_ens)

    #     print("\nEnsemble Performance Metrics:")
    #     print("R2 Score: {:.4f} ".format(r2_score(y_test, y_ens)))
    #     print(f"R2 Score: {r2_ens:.4f} ")
    #     print(f"MAE: {mae_ens:.4f}")
    #     print(f"RMSE: {rmse_ens:.4f}")

    # core = {"id": idx, "composition": comp, "target": y_test}
    # results = {f"pred-{num}": val for (num, val) in enumerate(y_ensemble)}
    # if model.robust:
    #     raise NotImplementedError(
    #     "Robust/Hetroskedastic classifaction is not currently implemented"
    #     )

    # df = pd.DataFrame({**core, **results})

    # if ensemble_folds == 1:
    #     df.to_csv(
    #         index=False,
    #         path_or_buf=(f"results/test_results_{model_name}_r-{run_id}.csv"),
    #     )
    # else:
    #     df.to_csv(
    #         index=False, path_or_buf=(f"results/ensemble_results_{model_name}.csv")
    #     )


if __name__ == "__main__":
    args = input_parser()

    print(f"The model will run on the {args.device} device")

    main(**vars(args))
