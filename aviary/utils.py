import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from datetime import datetime
from pickle import PickleError
from types import ModuleType
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import wandb
from pymatgen.core import Element
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from torch import LongTensor, Tensor
from torch.nn import CrossEntropyLoss, Embedding, L1Loss, MSELoss, NLLLoss
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from aviary import PKG_DIR, ROOT
from aviary.core import BaseModelClass, Normalizer, TaskType, sampled_softmax
from aviary.data import InMemoryDataLoader
from aviary.losses import robust_l1_loss, robust_l2_loss


def initialize_model(
    model_class: BaseModelClass,
    model_params: dict[str, Any],
    device: type[torch.device] | Literal["cuda", "cpu"],
    resume: str | None = None,
    fine_tune: str | None = None,
    transfer: str | None = None,
    **kwargs,
) -> BaseModelClass:
    """Initialise a model.

    Args:
        model_class (BaseModelClass): Which model class to initialize.
        model_params (dict[str, Any]): Dictionary containing model specific hyperparams.
        device (type[torch.device] | "cuda" | "cpu"): Device the model will run on.
        resume (str, optional): Path to model checkpoint to resume. Defaults to None.
        fine_tune (str, optional): Path to model checkpoint to fine tune. Defaults to
            None.
        transfer (str, optional): Path to model checkpoint to transfer. Defaults to
            None.
        **kwargs: Additional keyword arguments (will be ignored).

    Returns:
        BaseModelClass: An initialized model of type model_class.
    """
    robust = model_params["robust"]
    n_targets = model_params["n_targets"]

    if fine_tune is not None:
        print(f"Use material_nn and output_nn from {fine_tune=} as a starting point")
        checkpoint = torch.load(fine_tune, map_location=device, weights_only=False)

        # update the task disk to fine tuning task
        checkpoint["model_params"]["task_dict"] = model_params["task_dict"]

        model = model_class(**checkpoint["model_params"])
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        if model.model_params["robust"] != robust:
            raise ValueError(
                "cannot fine-tune between tasks with different numbers of outputs"
                " - use transfer option instead"
            )
        loaded_n_targets = model.model_params["n_targets"]
        if loaded_n_targets != n_targets:
            raise ValueError(
                f"n_targets mismatch between model_params dict ({n_targets}) and "
                f"loaded state dict ({loaded_n_targets})"
            )

    elif transfer is not None:
        # TODO rewrite/remove transfer option as it is not used/doesn't work as detailed
        print(
            f"Use material_nn from {transfer=} as a starting point and "
            "train the output_nn from scratch"
        )
        checkpoint = torch.load(transfer, map_location=device, weights_only=False)

        model = model_class(**model_params)
        model.to(device)

        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["state_dict"].items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    elif resume:
        print(f"Resuming training from {resume=}")
        checkpoint = torch.load(resume, map_location=device, weights_only=False)

        model = model_class(**checkpoint["model_params"])
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.epoch = checkpoint["epoch"]
        model.best_val_score = checkpoint["best_val_score"]

    else:
        model = model_class(**model_params)

        model.to(device)

    print(f"Total Number of Trainable Parameters: {model.num_params:,}")

    # TODO parallelise the code over multiple GPUs. Currently DataParallel
    # crashes as subsets of the batch have different sizes due to the use of
    # lists of lists rather the zero-padding.
    # if (torch.cuda.device_count() > 1) and (device==torch.device("cuda")):
    #     print("The model will use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model.to(device)

    return model


def initialize_optim(
    model: BaseModelClass,
    optim: type[Optimizer] | Literal["SGD", "Adam", "AdamW"],
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    device: type[torch.device] | Literal["cuda", "cpu"],
    milestones: Iterable = (),
    gamma: float = 0.3,
    resume: str | None = None,
    **kwargs,
) -> tuple[Optimizer, _LRScheduler]:
    """Initialize Optimizer and Scheduler.

    Args:
        model (BaseModelClass): Model to be optimized.
        optim (type[Optimizer] | "SGD" | "Adam" | "AdamW"): Which optimizer to use
        learning_rate (float): Learning rate for optimization
        weight_decay (float): Weight decay for optimizer
        momentum (float): Momentum for optimizer
        device (type[torch.device] | "cuda" | "cpu"): Device the model will run on
        milestones (Iterable, optional): When to decay learning rate. Defaults to ().
        gamma (float, optional): Multiplier for learning rate decay. Defaults to 0.3.
        resume (str, optional): Path to model checkpoint to resume. Defaults to None.
        **kwargs: Additional keyword arguments (will be ignored).


    Returns:
        tuple[Optimizer, _LRScheduler]: Optimizer and scheduler for given model
    """
    # Select optimizer
    optimizer: Optimizer
    if optim == "SGD":
        optimizer = SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optim == "Adam":
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optim == "AdamW":
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NameError("Only SGD, Adam or AdamW are allowed as --optim")

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if resume:
        # TODO work out how to ensure that we are using the same optimizer
        # when resuming such that the state dictionaries do not clash.
        # TODO breaking the function apart means we load the checkpoint twice.
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    return optimizer, scheduler


def initialize_losses(
    task_dict: dict[str, TaskType],
    loss_name_dict: dict[str, Literal["L1", "L2", "CSE"]],
    robust: bool = False,
) -> dict[str, tuple[TaskType, Callable]]:
    """Create a dictionary of loss functions for a multi-task model.

    Args:
        task_dict (dict[str, TaskType]): Map of target names to "regression" or
            "classification".
        loss_name_dict (dict[str, "L1" | "L2" | "CSE"]): Map of target names to loss
            type.
        robust (bool, optional): Whether to use an uncertainty adjusted loss. Defaults
            to False.

    Returns:
        dict[str, tuple[str, type[torch.nn.Module]]]: Dictionary of losses for each task
    """
    loss_func_dict: dict[str, tuple[TaskType, Callable]] = {}
    for name, task in task_dict.items():
        # Select Task and Loss Function
        if task == "classification":
            if loss_name_dict[name] != "CSE":
                raise NameError("Only CSE loss allowed for classification tasks")

            if robust:
                loss_func_dict[name] = (task, NLLLoss())
            else:
                loss_func_dict[name] = (task, CrossEntropyLoss())

        elif task == "regression":
            if robust:
                if loss_name_dict[name] == "L1":
                    loss_func_dict[name] = (task, robust_l1_loss)
                elif loss_name_dict[name] == "L2":
                    loss_func_dict[name] = (task, robust_l2_loss)
                else:
                    raise NameError(
                        "Only L1 or L2 losses are allowed for robust regression tasks"
                    )
            elif loss_name_dict[name] == "L1":
                loss_func_dict[name] = (task, L1Loss())
            elif loss_name_dict[name] == "L2":
                loss_func_dict[name] = (task, MSELoss())
            else:
                raise NameError("Only L1 or L2 losses are allowed for regression tasks")

    return loss_func_dict


def init_normalizers(
    task_dict: dict[str, TaskType],
    device: type[torch.device] | Literal["cuda", "cpu"],
    resume: str | None = None,
) -> dict[str, Normalizer | None]:
    """Initialise a Normalizer to scale the output targets.

    Args:
        task_dict (dict[str, TaskType]): Map of target names to "regression" or
            "classification".
        device (torch.device | "cuda" | "cpu"): Device the model will run on
        resume (str, optional): Path to model checkpoint to resume. Defaults to None.

    Returns:
        dict[str, Normalizer]: Dictionary of Normalizers for each task
    """
    normalizer_dict: dict[str, Normalizer | None] = {}
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        for task, state_dict in checkpoint["normalizer_dict"].items():
            normalizer_dict[task] = Normalizer.from_state_dict(state_dict)

        return normalizer_dict

    for target, task in task_dict.items():
        # Select Task and Loss Function
        if task == "regression":
            normalizer_dict[target] = Normalizer()
        else:
            normalizer_dict[target] = None

    return normalizer_dict


def train_ensemble(
    model_class: BaseModelClass,
    model_name: str,
    run_id: int,
    ensemble_folds: int,
    epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    log: Literal["tensorboard", "wandb"],
    setup_params: dict[str, Any],
    restart_params: dict[str, Any],
    model_params: dict[str, Any],
    loss_dict: dict[str, Literal["L1", "L2", "CSE"]],
    patience: int | None = None,
    verbose: bool = False,
) -> None:
    """Train multiple models that form an ensemble in serial with this convenience
    function.

    Args:
        model_class (BaseModelClass): Which model class to initialize.
        model_name (str): String describing the model.
        run_id (int): Unique identifier of the model run.
        ensemble_folds (int): Number of members in ensemble.
        epochs (int): Number of epochs to train for.
        train_loader (DataLoader): Dataloader containing training data.
        val_loader (DataLoader | None): Dataloader containing validation data.
        log (bool): Whether to log intermediate metrics to TensorBoard.
        setup_params (dict[str, Any]): Dictionary of setup parameters
        restart_params (dict[str, Any]): Dictionary of restart parameters
        model_params (dict[str, Any]): Dictionary of model parameters
        loss_dict (dict[str, "L1" | "L2" | "CSE"]): Map of target names
            to loss functions.
        patience (int, optional): Maximum number of epochs without improvement
            when early stopping. Defaults to None.
        verbose (bool, optional): Whether to show progress bars for each epoch.
    """
    #  this allows us to run ensembles in parallel rather than in series
    #  by specifying the run-id arg.
    for r_id in [run_id] if ensemble_folds == 1 else range(ensemble_folds):
        model = initialize_model(
            model_class=model_class,
            model_params=model_params,
            **setup_params,
            **restart_params,
        )
        optimizer, scheduler = initialize_optim(
            model,
            **setup_params,
            **restart_params,
        )

        loss_func_dict = initialize_losses(
            model.task_dict, loss_dict, model_params["robust"]
        )
        normalizer_dict = init_normalizers(
            model.task_dict, setup_params["device"], restart_params["resume"]
        )

        for target, normalizer in normalizer_dict.items():
            if normalizer is not None:
                if isinstance(train_loader, InMemoryDataLoader):
                    # FIXME: this is really brittle but it works for now.
                    sample_target = train_loader.tensors[1]
                else:
                    data = train_loader.dataset
                    if isinstance(train_loader.dataset, Subset):
                        sample_target = Tensor(
                            data.dataset.df[target].iloc[data.indices].to_numpy()
                        )
                    else:
                        sample_target = Tensor(data.df[target].to_numpy())

                if not restart_params["resume"]:
                    normalizer.fit(sample_target)
                print(f"Dummy MAE: {(sample_target - normalizer.mean).abs().mean():.4f}")

        if log == "tensorboard":
            writer = SummaryWriter(
                f"{ROOT}/runs/{model_name}/{model_name}-r{r_id}_{datetime.now():%d-%m-%Y_%H-%M-%S}"
            )
        elif log == "wandb":
            wandb.init(
                project="lightning_logs",
                # https://docs.wandb.ai/guides/track/launch#init-start-error
                settings=wandb.Settings(start_method="fork"),
                name=f"{model_name}-r{r_id}",
                config={
                    "model_params": model_params,
                    "setup_params": setup_params,
                    "restart_params": restart_params,
                    "loss_dict": loss_dict,
                },
            )
            writer = "wandb"
        else:
            writer = None

        if (val_loader is not None) and (model.best_val_scores is None):
            print("Getting Validation Baseline")
            with torch.no_grad():
                v_metrics = model.evaluate(
                    val_loader,
                    loss_dict=loss_func_dict,
                    optimizer=None,
                    normalizer_dict=normalizer_dict,
                    action="evaluate",
                    verbose=verbose,
                )

                val_score = {}

                for name, task in model.task_dict.items():
                    if task == "regression":
                        MAE = val_score[name] = v_metrics[name]["MAE"]
                        print(f"Validation Baseline - {name}: {MAE=:.2f}")
                    elif task == "classification":
                        Accuracy = val_score[name] = v_metrics[name]["Accuracy"]
                        print(f"Validation Baseline - {name}: {Accuracy=:.2f}")
                model.best_val_scores = val_score

        model.fit(
            train_loader,
            val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            loss_dict=loss_func_dict,
            normalizer_dict=normalizer_dict,
            model_name=model_name,
            run_id=r_id,
            writer=writer,
            patience=patience,
        )


@torch.no_grad()
def results_multitask(
    model_class: BaseModelClass,
    model_name: str,
    run_id: int,
    ensemble_folds: int,
    test_loader: DataLoader | InMemoryDataLoader,
    robust: bool,
    task_dict: dict[str, TaskType],
    device: type[torch.device] | Literal["cuda", "cpu"],
    eval_type: str = "checkpoint",
    print_results: bool = True,
    save_results: bool = True,
) -> dict[str, dict[str, list | np.ndarray]]:
    """Take an ensemble of models and evaluate their performance on the test set.

    Args:
        model_class (BaseModelClass): Which model class to initialize.
        model_name (str): String describing the model.
        run_id (int): Unique identifier of the model run.
        ensemble_folds (int): Number of members in ensemble.
        test_loader (DataLoader): Dataloader containing testing data.
        robust (bool): Whether to estimate standard deviation for use in a robust
            loss function.
        task_dict (dict[str, TaskType]): Map of target names to "regression" or
            "classification".
        device (type[torch.device] | "cuda" | "cpu"): Device the model will run on
        eval_type (str, optional): Whether to use final or early-stopping checkpoints.
            Defaults to "checkpoint".
        print_results (bool, optional): Whether to print out summary metrics.
            Defaults to True.
        save_results (bool, optional): Whether to save results dict. Defaults to True.

    Returns:
        dict[str, dict[str, list | np.array]]: Dictionary of predicted results for each
            task.
    """
    if not (print_results or save_results):
        raise ValueError(
            "Evaluating Model pointless if both 'print_results' and "
            "'save_results' are False."
        )

    print(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        "------------Evaluate model on Test Set------------\n"
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    )
    results_dict: dict[str, dict[str, list | np.ndarray]] = {}
    n_test = (
        len(test_loader.tensors[0])
        if isinstance(test_loader, InMemoryDataLoader)
        else len(test_loader.dataset)
    )
    for target_name, task_type in task_dict.items():
        results_dict[target_name] = defaultdict(
            list
            if task_type == "classification"
            else lambda: np.zeros((ensemble_folds, n_test))  # type: ignore[call-overload]
        )

    for ens_idx in range(ensemble_folds):
        if ensemble_folds == 1:
            resume = f"{ROOT}/models/{model_name}/{eval_type}-r{run_id}.pth.tar"
            print("Evaluating Model")
        else:
            resume = f"{ROOT}/models/{model_name}/{eval_type}-r{ens_idx}.pth.tar"
            print(f"Evaluating Model {ens_idx + 1}/{ensemble_folds}")

        checkpoint = torch.load(resume, map_location=device, weights_only=False)

        if checkpoint["model_params"]["robust"] != robust:
            raise ValueError(f"robustness of checkpoint {resume=} is not {robust}")

        chkpt_task_dict = checkpoint["model_params"]["task_dict"]
        if chkpt_task_dict != task_dict:
            raise ValueError(
                f"task_dict {chkpt_task_dict} of checkpoint {resume=} does not match "
                f"provided {task_dict=}"
            )

        model: BaseModelClass = model_class(**checkpoint["model_params"])
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        normalizer_dict: dict[str, Normalizer | None] = {}
        for task_type, state_dict in checkpoint["normalizer_dict"].items():
            if state_dict is not None:
                normalizer_dict[task_type] = Normalizer.from_state_dict(state_dict)
            else:
                normalizer_dict[task_type] = None

        y_test, outputs, *ids = model.predict(test_loader)

        for output, targets, (target_name, task_type), res_dict in zip(
            outputs, y_test, model.task_dict.items(), results_dict.values(), strict=False
        ):
            if task_type == "regression":
                normalizer = normalizer_dict[target_name]
                assert isinstance(normalizer, Normalizer)
                if model.robust:
                    mean, log_std = output.unbind(dim=1)
                    preds = normalizer.denorm(mean.data.cpu())
                    ale_std = torch.exp(log_std).data.cpu() * normalizer.std
                    res_dict["ale"][ens_idx, :] = ale_std.view(-1).numpy()  # type: ignore[call-overload]
                else:
                    preds = normalizer.denorm(output.data.cpu())

                res_dict["preds"][ens_idx, :] = preds.view(-1).numpy()  # type: ignore[call-overload]

            elif task_type == "classification":
                if model.robust:
                    mean, log_std = output.chunk(2, dim=1)
                    logits = sampled_softmax(mean, log_std, samples=10).data.cpu().numpy()
                    pre_logits = mean.data.cpu().numpy()
                    pre_logits_std = torch.exp(log_std).data.cpu().numpy()
                    res_dict["pre-logits_ale"].append(pre_logits_std)  # type: ignore[union-attr]
                else:
                    pre_logits = output.data.cpu().numpy()
                    logits = pre_logits.softmax(1)

                res_dict["pre-logits"].append(pre_logits)  # type: ignore[union-attr]
                res_dict["logits"].append(logits)  # type: ignore[union-attr]

            res_dict["targets"] = targets

    if save_results:
        identifier_names = (
            [f"idx_{i}" for i in range(len(ids))]
            if isinstance(test_loader, InMemoryDataLoader)
            else test_loader.dataset.dataset.identifiers
        )
        save_results_dict(
            dict(zip(identifier_names, *ids, strict=False)),
            results_dict,
            model_name,
            f"-r{run_id}",
        )

    if print_results:
        for target_name, task_type in task_dict.items():
            print(f"\nTask: {target_name=} on test set")
            if task_type == "regression":
                print_metrics_regression(**results_dict[target_name])  # type: ignore[arg-type]
            elif task_type == "classification":
                print_metrics_classification(**results_dict[target_name])  # type: ignore[arg-type]

    return results_dict


def print_metrics_regression(targets: np.ndarray, preds: np.ndarray, **kwargs) -> None:
    """Print out single model and/or ensemble metrics for a regression task.

    Args:
        targets (np.array): Targets for regression task. Shape (n_test,).
        preds (np.array): Model predictions. Shape (n_ensemble, n_test).
        kwargs: unused entries from the results dictionary.
    """
    ensemble_folds = preds.shape[0]
    res = preds - targets
    mae = np.mean(np.abs(res), axis=1)
    mse = np.mean(np.square(res), axis=1)
    rmse = np.sqrt(mse)
    r2 = r2_score(
        np.repeat(targets[:, None], ensemble_folds, axis=1),
        preds.T,
        multioutput="raw_values",
    )

    r2_avg = np.mean(r2)
    r2_std = np.std(r2)

    mae_avg = np.mean(mae)
    mae_std = np.std(mae) / np.sqrt(mae.shape[0])

    rmse_avg = np.mean(rmse)
    rmse_std = np.std(rmse) / np.sqrt(rmse.shape[0])

    if ensemble_folds == 1:
        print("Model Performance Metrics:")
        print(f"R2 Score: {r2_avg:.4f} ")
        print(f"MAE: {mae_avg:.4f}")
        print(f"RMSE: {rmse_avg:.4f}")
    else:
        print("Model Performance Metrics:")
        print(f"R2 Score: {r2_avg:.4f} +/- {r2_std:.4f}")
        print(f"MAE: {mae_avg:.4f} +/- {mae_std:.4f}")
        print(f"RMSE: {rmse_avg:.4f} +/- {rmse_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        y_ens = np.mean(preds, axis=0)

        mae_ens = np.abs(targets - y_ens).mean()
        mse_ens = np.square(targets - y_ens).mean()
        rmse_ens = np.sqrt(mse_ens)

        r2_ens = r2_score(targets, y_ens)

        print("\nEnsemble Performance Metrics:")
        print(f"R2 Score : {r2_ens:.4f} ")
        print(f"MAE  : {mae_ens:.4f}")
        print(f"RMSE : {rmse_ens:.4f}")


def print_metrics_classification(
    targets: LongTensor,
    logits: Tensor,
    average: Literal["micro", "macro", "samples", "weighted"] = "micro",
    **kwargs,
) -> None:
    """Print out metrics for a classification task.

    TODO make less janky, first index is for ensembles, second data, third classes.
    always calculate metrics in the multi-class setting. How to convert binary labels
    to multi-task automatically?

    Args:
        targets (np.array): Categorical encoding of the tasks. Shape (n_test,).
        logits (list[n_ens * np.array(n_targets, n_test)]): logits predicted by the
            model.
        average ("micro" | "macro" | "samples" | "weighted"): Determines the type of
            data averaging. Defaults to 'micro' which calculates metrics globally by
            considering each element of the label indicator matrix as a label.
        kwargs: unused entries from the results dictionary
    """
    logits = np.asarray(logits)
    if len(logits.shape) != 3:
        raise ValueError(
            "please insure that the logits are of the form (n_ens, n_data, n_classes)"
        )

    acc = np.zeros(len(logits))
    roc_auc = np.zeros(len(logits))
    precision = np.zeros(len(logits))
    recall = np.zeros(len(logits))
    fscore = np.zeros(len(logits))

    target_ohe = np.zeros_like(logits[0])
    target_ohe[np.arange(targets.size), targets] = 1

    for j, y_logit in enumerate(logits):
        y_pred = np.argmax(y_logit, axis=1)

        acc[j] = accuracy_score(targets, y_pred)
        roc_auc[j] = roc_auc_score(target_ohe, y_logit, average=average)
        precision[j], recall[j], fscore[j], _ = precision_recall_fscore_support(
            targets, y_pred, average=average
        )

    if len(logits) == 1:
        print("\nModel Performance Metrics:")
        print(f"Accuracy : {acc[0]:.4f} ")
        print(f"ROC-AUC  : {roc_auc[0]:.4f}")
        print(f"Weighted Precision : {precision[0]:.4f}")
        print(f"Weighted Recall    : {recall[0]:.4f}")
        print(f"Weighted F-score   : {fscore[0]:.4f}")
    else:
        acc_avg = np.mean(acc)
        acc_std = np.std(acc) / np.sqrt(acc.shape[0])

        roc_auc_avg = np.mean(roc_auc)
        roc_auc_std = np.std(roc_auc) / np.sqrt(roc_auc.shape[0])

        prec_avg = np.mean(precision)
        prec_std = np.std(precision) / np.sqrt(precision.shape[0])

        recall_avg = np.mean(recall)
        recall_std = np.std(recall) / np.sqrt(recall.shape[0])

        fscore_avg = np.mean(fscore)
        fscore_std = np.std(fscore) / np.sqrt(fscore.shape[0])

        print("\nModel Performance Metrics:")
        print(f"Accuracy : {acc_avg:.4f} +/- {acc_std:.4f}")
        print(f"ROC-AUC  : {roc_auc_avg:.4f} +/- {roc_auc_std:.4f}")
        print(f"Weighted Precision : {prec_avg:.4f} +/- {prec_std:.4f}")
        print(f"Weighted Recall    : {recall_avg:.4f} +/- {recall_std:.4f}")
        print(f"Weighted F-score   : {fscore_avg:.4f} +/- {fscore_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        ens_logits = np.mean(logits, axis=0)

        y_pred = np.argmax(ens_logits, axis=1)

        ens_acc = accuracy_score(targets, y_pred)
        ens_roc_auc = roc_auc_score(target_ohe, ens_logits, average=average)
        ens_prec, ens_recall, ens_fscore, _ = precision_recall_fscore_support(
            targets, y_pred, average=average
        )

        print("\nEnsemble Performance Metrics:")
        print(f"Accuracy : {ens_acc:.4f} ")
        print(f"ROC-AUC  : {ens_roc_auc:.4f}")
        print(f"Weighted Precision : {ens_prec:.4f}")
        print(f"Weighted Recall    : {ens_recall:.4f}")
        print(f"Weighted F-score   : {ens_fscore:.4f}")


def save_results_dict(
    ids: dict[str, list[str | int]],
    results_dict: dict[str, Any],
    model_name: str,
    run_id: str | None = None,
) -> None:
    """Save the results to a file after model evaluation.

    Args:
        ids (dict[str, list[str | int]]): Each key is the name of an identifier
            (e.g. material ID, composition, ...) and its value a list of IDs.
        results_dict (dict[str, Any]): nested dictionary of results {name: {col: data}}
        model_name (str): The name given the model via the --model-name flag.
        run_id (str): The run ID given to the model via the --run-id flag.
    """
    results: dict[str, np.ndarray] = {}

    for target_name, target_data in results_dict.items():
        for col, data in target_data.items():
            # NOTE we save pre_logits rather than logits due to fact
            # that with the heteroskedastic setup we want to be able to
            # sample from the Gaussian distributed pre_logits we parameterize.
            if "pre-logits" in col:
                for n_ens, y_pre_logit in enumerate(data):
                    results |= {
                        f"{target_name}_{col}_c{lab}_n{n_ens}": val.ravel()
                        for lab, val in enumerate(y_pre_logit.T)
                    }

            elif "pred" in col or "ale" in col:
                # elif so that pre-logit-ale doesn't trigger
                results |= {
                    f"{target_name}_{col}_n{n_ens}": val.ravel()
                    for (n_ens, val) in enumerate(data)
                }

            elif col == "target":
                results |= {f"{target_name}_target": data}

    df = pd.DataFrame({**ids, **results})

    os.makedirs("results", exist_ok=True)

    csv_path = f"results/{model_name.replace('/', '_') + (run_id or '')}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved model predictions to {csv_path!r}")


def get_metrics(
    targets: np.ndarray | pd.Series,
    predictions: np.ndarray | pd.Series,
    type: TaskType,
    prec: int = 4,
) -> dict:
    """Get performance metrics for model predictions.

    Args:
        targets (np.array): Ground truth values.
        predictions (np.array): Model predictions. Should be class probabilities for
            classification (i.e. output model after applying softmax/sigmoid). Same
            shape as targets for regression, and [len(targets), n_classes] for
            classification.
        type ('regression' | 'classification'): Task type.
        prec (int, optional): Decimal places to round metrics to. Defaults to 4.

    Returns:
        dict[str, float]: Keys are rmse, mae, r2 for regression and accuracy,
            balanced_accuracy, f1, rocauc for classification.
    """
    metrics = {}
    nans = np.isnan(np.column_stack([targets, predictions])).any(axis=1)
    # r2_score() and roc_auc_score() don't auto-handle NaNs
    targets, predictions = targets[~nans], predictions[~nans]

    if type == "regression":
        metrics["MAE"] = np.abs(targets - predictions).mean()
        metrics["RMSE"] = ((targets - predictions) ** 2).mean() ** 0.5
        metrics["R2"] = r2_score(targets, predictions)
    elif type == "classification":
        pred_labels = predictions.argmax(axis=1)

        metrics["accuracy"] = accuracy_score(targets, pred_labels)
        metrics["balanced_accuracy"] = balanced_accuracy_score(targets, pred_labels)
        metrics["F1"] = f1_score(targets, pred_labels)
        class1_probas = predictions[:, 1]
        metrics["ROCAUC"] = roc_auc_score(targets, class1_probas)
    else:
        raise ValueError(f"Invalid task type: {type}")

    return {key: round(float(val), prec) for key, val in metrics.items()}


def get_element_embedding(elem_embedding: str) -> Embedding:
    """Get an element embedding from a file.

    Args:
        elem_embedding (str): The path to the element embedding file.

    Returns:
        Embedding: The element embedding.
    """
    if os.path.isfile(elem_embedding):
        pass
    elif elem_embedding in ["matscholar200", "cgcnn92", "megnet16", "onehot112"]:
        elem_embedding = f"{PKG_DIR}/embeddings/element/{elem_embedding}.json"
    else:
        raise ValueError(f"Invalid element embedding: {elem_embedding}")

    with open(elem_embedding) as file:
        elem_features = json.load(file)

    max_z = max(Element(elem).Z for elem in elem_features)
    elem_emb_len = len(next(iter(elem_features.values())))
    elem_feature_matrix = torch.zeros((max_z + 1, elem_emb_len))
    for elem, feature in elem_features.items():
        elem_feature_matrix[Element(elem).Z] = torch.tensor(feature)

    embedding = Embedding(max_z + 1, elem_emb_len)
    embedding.weight.data.copy_(elem_feature_matrix)

    return embedding


def get_sym_embedding(sym_embedding: str) -> Embedding:
    """Get a symmetry embedding from a file.

    Args:
        sym_embedding (str): The path to the symmetry embedding file.

    Returns:
        Embedding: The symmetry embedding.
    """
    if os.path.isfile(sym_embedding):
        pass
    elif sym_embedding in ("bra-alg-off", "spg-alg-off"):
        sym_embedding = f"{PKG_DIR}/embeddings/wyckoff/{sym_embedding}.json"
    else:
        raise ValueError(f"Invalid symmetry embedding: {sym_embedding}")

    with open(sym_embedding) as sym_file:
        sym_features = json.load(sym_file)

    sym_emb_len = len(next(iter(next(iter(sym_features.values())).values())))

    len_sym_features = sum(len(feature) for feature in sym_features.values())
    sym_feature_matrix = torch.zeros((len_sym_features, sym_emb_len))
    sym_idx = 0
    for embeddings in sym_features.values():
        for feature in embeddings.values():
            sym_feature_matrix[sym_idx] = torch.tensor(feature)
            sym_idx += 1

    embedding = Embedding(len_sym_features, sym_emb_len)
    embedding.weight.data.copy_(sym_feature_matrix)

    return embedding


def as_dict_handler(obj: Any) -> dict[str, Any] | None:
    """Pass this func as json.dump(handler=) or as pandas.to_json(default_handler=)."""
    try:
        return obj.as_dict()  # all MSONable objects implement as_dict()
    except AttributeError:
        return None  # replace unhandled objects with None in serialized data


def update_module_path_in_pickled_object(
    pickle_path: str, old_module_path: str, new_module: ModuleType
) -> None:
    """Update a python module's dotted path in a pickle dump if the
    corresponding file was renamed.

    Implements the advice in https://stackoverflow.com/a/2121918.
    Posted at https://stackoverflow.com/a/73696259.

    Args:
        pickle_path (str): Path to the pickled object.
        old_module_path (str): The old.dotted.path.to.renamed.module.
        new_module (ModuleType): from new.location import module.
    """
    if not os.path.isfile(pickle_path):
        raise FileNotFoundError(pickle_path)

    sys.modules[old_module_path] = new_module

    try:
        dic = torch.load(pickle_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise PickleError(pickle_path) from exc

    del sys.modules[old_module_path]

    torch.save(dic, pickle_path)


@contextmanager
def print_walltime(
    start_desc: str = "",
    end_desc: str = "",
    newline: bool = True,
    min_run_time: float = 1,
) -> Generator[None, None, None]:
    """Context manager and decorator that prints the wall time of its lifetime.

    Args:
        start_desc (str): Text to print when entering context. Defaults to ''.
        end_desc (str): Text to print when exiting context. Will be followed by 'took
            {duration} sec'. i.e. f"{end_desc} took 1.23 sec". Defaults to ''.
        newline (bool): Whether to print a newline after start_desc. Defaults to True.
        min_run_time (float): Minimum wall time in seconds below which nothing will be
            printed. Defaults to 1.
    """
    start_time = time.perf_counter()
    if start_desc:
        print(start_desc, end="\n" if newline else "")

    try:
        yield
    finally:
        run_time = time.perf_counter() - start_time
        # don't print on immediate failures
        if run_time > min_run_time:
            print(f"{end_desc} took {run_time:.2f} sec")
