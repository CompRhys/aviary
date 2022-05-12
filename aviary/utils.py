from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from torch import LongTensor, Tensor
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, NLLLoss
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from aviary import ROOT
from aviary.core import BaseModelClass, Normalizer, TaskType, sampled_softmax
from aviary.losses import RobustL1Loss, RobustL2Loss

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


def init_model(
    model_class: type[BaseModelClass],
    model_params: dict[str, Any],
    device: type[torch.device] | Literal["cuda", "cpu"],
    resume: str = None,
    fine_tune: str = None,
    transfer: str = None,
    **kwargs,
) -> type[BaseModelClass]:
    """Initialise a model

    Args:
        model_class (type[BaseModelClass]): Which model class to initialize.
        model_params (dict[str, Any]): Dictionary containing model specific hyperparameters.
        device (type[torch.device] | "cuda" | "cpu"): Device the model will run on.
        resume (str, optional): Path to model checkpoint to resume. Defaults to None.
        fine_tune (str, optional): Path to model checkpoint to fine tune. Defaults to None.
        transfer (str, optional): Path to model checkpoint to transfer. Defaults to None.

    Returns:
        BaseModelClass: An initialised model of type model_class.
    """
    robust = model_params["robust"]
    n_targets = model_params["n_targets"]

    if fine_tune is not None:
        print(f"Use material_nn and output_nn from '{fine_tune}' as a starting point")
        checkpoint = torch.load(fine_tune, map_location=device)

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
                f"n_targets mismatch between model_params dict ({n_targets}) and loaded "
                f"state dict ({loaded_n_targets})"
            )

    elif transfer is not None:
        # TODO rewrite/remove transfer option as it is not used/doesn't work as detailed
        print(
            f"Use material_nn from '{transfer}' as a starting point and "
            "train the output_nn from scratch"
        )
        checkpoint = torch.load(transfer, map_location=device)

        model = model_class(**model_params)
        model.to(device)

        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["state_dict"].items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    elif resume:
        print(f"Resuming training from '{resume}'")
        checkpoint = torch.load(resume, map_location=device)

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


def init_optim(
    model: type[BaseModelClass],
    optim: type[Optimizer] | Literal["SGD", "Adam", "AdamW"],
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    device: type[torch.device] | Literal["cuda", "cpu"],
    milestones: Iterable = (),
    gamma: float = 0.3,
    resume: str = None,
    **kwargs,
) -> tuple[Optimizer, _LRScheduler]:
    """Initialize Optimizer and Scheduler.

    Args:
        model (type[BaseModelClass]): Model to be optimized.
        optim (type[Optimizer] | "SGD" | "Adam" | "AdamW"): Which optimizer to use
        learning_rate (float): Learning rate for optimization
        weight_decay (float): Weight decay for optimizer
        momentum (float): Momentum for optimizer
        device (type[torch.device] | "cuda" | "cpu"): Device the model will run on
        milestones (Iterable, optional): When to decay learning rate. Defaults to ().
        gamma (float, optional): Multiplier for learning rate decay. Defaults to 0.3.
        resume (str, optional): Path to model checkpoint to resume. Defaults to None.


    Returns:
        tuple[Optimizer, _LRScheduler]: Optimizer and scheduler for given model
    """
    # Select Optimiser
    if optim == "SGD":
        optimizer = SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    elif optim == "Adam":
        optimizer = Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optim == "AdamW":
        optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise NameError("Only SGD, Adam or AdamW are allowed as --optim")

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if resume:
        # TODO work out how to ensure that we are using the same optimizer
        # when resuming such that the state dictionaries do not clash.
        # TODO breaking the function apart means we load the checkpoint twice.
        checkpoint = torch.load(resume, map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    return optimizer, scheduler


def init_losses(
    task_dict: dict[str, TaskType],
    loss_dict: dict[str, Literal["L1", "L2", "CSE"]],
    robust: bool = False,
) -> dict[str, tuple[str, type[torch.nn.Module]]]:
    """_summary_

    Args:
        task_dict (dict[str, TaskType]): Map of target names to "regression" or "classification".
        loss_dict (dict[str, "L1" | "L2" | "CSE"]): Map of target names to loss functions.
        robust (bool, optional): Whether to use an uncertainty adjusted loss. Defaults to False.

    Returns:
        dict[str, tuple[str, type[torch.nn.Module]]]: Dictionary of losses for each task
    """
    criterion_dict: dict[str, tuple[str, type[torch.nn.Module]]] = {}
    for name, task in task_dict.items():
        # Select Task and Loss Function
        if task == "classification":
            if loss_dict[name] != "CSE":
                raise NameError("Only CSE loss allowed for classification tasks")

            if robust:
                criterion_dict[name] = (task, NLLLoss())
            else:
                criterion_dict[name] = (task, CrossEntropyLoss())

        elif task == "regression":
            if robust:
                if loss_dict[name] == "L1":
                    criterion_dict[name] = (task, RobustL1Loss)
                elif loss_dict[name] == "L2":
                    criterion_dict[name] = (task, RobustL2Loss)
                else:
                    raise NameError(
                        "Only L1 or L2 losses are allowed for robust regression tasks"
                    )
            else:
                if loss_dict[name] == "L1":
                    criterion_dict[name] = (task, L1Loss())
                elif loss_dict[name] == "L2":
                    criterion_dict[name] = (task, MSELoss())
                else:
                    raise NameError(
                        "Only L1 or L2 losses are allowed for regression tasks"
                    )

    return criterion_dict


def init_normalizers(
    task_dict: dict[str, TaskType],
    device: type[torch.device] | Literal["cuda", "cpu"],
    resume: str = None,
) -> dict[str, Normalizer]:
    """Initialise a Normalizer to scale the output targets

    Args:
        task_dict (dict[str, TaskType]): Map of target names to "regression" or "classification".
        device (torch.device | "cuda" | "cpu"): Device the model will run on
        resume (str, optional): Path to model checkpoint to resume. Defaults to None.

    Returns:
        dict[str, Normalizer]: Dictionary of Normalizers for each task
    """
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        normalizer_dict = {}
        for task, state_dict in checkpoint["normalizer_dict"].items():
            normalizer_dict[task] = Normalizer.from_state_dict(state_dict)

        return normalizer_dict

    normalizer_dict = {}
    for target, task in task_dict.items():
        # Select Task and Loss Function
        if task == "regression":
            normalizer_dict[target] = Normalizer()
        else:
            normalizer_dict[target] = None

    return normalizer_dict


def train_ensemble(
    model_class: type[BaseModelClass],
    model_name: str,
    run_id: int,
    ensemble_folds: int,
    epochs: int,
    train_set: Dataset | Subset,
    val_set: Dataset | Subset,
    log: bool,
    data_params: dict[str, Any],
    setup_params: dict[str, Any],
    restart_params: dict[str, Any],
    model_params: dict[str, Any],
    loss_dict: dict[str, Literal["L1", "L2", "CSE"]],
    patience: int = None,
    verbose: bool = False,
) -> None:
    """Convenience method to train multiple models in serial.

    Args:
        model_class (type[BaseModelClass]): Which model class to initialize.
        model_name (str): String describing the model.
        run_id (int): Unique identifier of the model run.
        ensemble_folds (int): Number of members in ensemble.
        epochs (int): Number of epochs to train for.
        train_set (Subset): Dataloader containing training data.
        val_set (Subset): Dataloader containing validation data.
        log (bool): Whether to log intermediate metrics to tensorboard.
        data_params (dict[str, Any]): Dictionary of dataloader parameters
        setup_params (dict[str, Any]): Dictionary of setup parameters
        restart_params (dict[str, Any]): Dictionary of restart parameters
        model_params (dict[str, Any]): Dictionary of model parameters
        loss_dict (dict[str, "L1" | "L2" | "CSE"]): Map of target names
            to loss functions.
        patience (int, optional): Maximum number of epochs without improvement
            when early stopping. Defaults to None.
        verbose (bool, optional): Whether to show progress bars for each epoch.
    """
    if isinstance(train_set, Subset):
        train_set = train_set.dataset
    if isinstance(val_set, Subset):
        val_set = val_set.dataset

    train_generator = DataLoader(train_set, **data_params)
    print(f"Training on {len(train_set):,} samples")

    if val_set is not None:
        data_params.update({"batch_size": 16 * data_params["batch_size"]})
        val_generator = DataLoader(val_set, **data_params)
    else:
        val_generator = None

    for j in range(ensemble_folds):
        #  this allows us to run ensembles in parallel rather than in series
        #  by specifying the run-id arg.
        if ensemble_folds == 1:
            j = run_id

        model = init_model(
            model_class=model_class,
            model_params=model_params,
            **setup_params,
            **restart_params,
        )
        optimizer, scheduler = init_optim(
            model,
            **setup_params,
            **restart_params,
        )

        criterion_dict = init_losses(model.task_dict, loss_dict, model_params["robust"])
        normalizer_dict = init_normalizers(
            model.task_dict, setup_params["device"], restart_params["resume"]
        )

        for target, normalizer in normalizer_dict.items():
            if normalizer is not None:
                sample_target = Tensor(train_set.df[target].values)
                if not restart_params["resume"]:
                    normalizer.fit(sample_target)
                print(
                    f"Dummy MAE: {(sample_target - normalizer.mean).abs().mean():.4f}"
                )

        if log:
            writer = SummaryWriter(
                f"{ROOT}/runs/{model_name}/{model_name}-r{j}_{datetime.now():%d-%m-%Y_%H-%M-%S}"
            )
        else:
            writer = None

        if (val_set is not None) and (model.best_val_scores is None):
            print("Getting Validation Baseline")
            with torch.no_grad():
                v_metrics = model.evaluate(
                    generator=val_generator,
                    criterion_dict=criterion_dict,
                    optimizer=None,
                    normalizer_dict=normalizer_dict,
                    action="val",
                    verbose=verbose,
                )

                val_score = {}

                for name, task in model.task_dict.items():
                    if task == "regression":
                        val_score[name] = v_metrics[name]["MAE"]
                        print(
                            f"Validation Baseline - {name}: MAE {val_score[name]:.2f}"
                        )
                    elif task == "classification":
                        val_score[name] = v_metrics[name]["Acc"]
                        print(
                            f"Validation Baseline - {name}: Acc {val_score[name]:.2f}"
                        )
                model.best_val_scores = val_score

        model.fit(
            train_generator=train_generator,
            val_generator=val_generator,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            criterion_dict=criterion_dict,
            normalizer_dict=normalizer_dict,
            model_name=model_name,
            run_id=j,
            writer=writer,
            patience=patience,
        )


# TODO find a better name for this function @janosh
@torch.no_grad()
def results_multitask(  # noqa: C901
    model_class: type[BaseModelClass],
    model_name: str,
    run_id: int,
    ensemble_folds: int,
    test_set: Dataset | Subset,
    data_params: dict[str, Any],
    robust: bool,
    task_dict: dict[str, TaskType],
    device: type[torch.device] | Literal["cuda", "cpu"],
    eval_type: str = "checkpoint",
    print_results: bool = True,
    save_results: bool = True,
) -> dict[str, dict[str, list | np.ndarray]]:
    """Take an ensemble of models and evaluate their performance on the test set.

    Args:
        model_name (str): String describing the model.
        run_id (int): Unique identifier of the model run.
        ensemble_folds (int): Number of members in ensemble.
        test_set (Subset): Dataloader containing testing data.
        data_params (dict[str, Any]): Dictionary of dataloader parameters
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
        dict[str, dict[str, list | np.ndarray]]: Dictionary of predicted results for each
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

    if isinstance(test_set, Subset):
        test_set = test_set.dataset

    test_generator = DataLoader(test_set, **data_params)
    print(f"Testing on {len(test_set):,} samples")

    results_dict: dict[str, dict[str, list | np.ndarray]] = {n: {} for n in task_dict}
    for name, task in task_dict.items():
        if task == "regression":
            results_dict[name]["pred"] = np.zeros((ensemble_folds, len(test_set)))
            if robust:
                results_dict[name]["ale"] = np.zeros((ensemble_folds, len(test_set)))

        elif task == "classification":
            results_dict[name]["logits"] = []
            results_dict[name]["pre-logits"] = []
            if robust:
                results_dict[name]["pre-logits_ale"] = []

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            resume = f"{ROOT}/models/{model_name}/{eval_type}-r{run_id}.pth.tar"
            print("Evaluating Model")
        else:
            resume = f"{ROOT}/models/{model_name}/{eval_type}-r{j}.pth.tar"
            print(f"Evaluating Model {j + 1}/{ensemble_folds}")

        if not os.path.isfile(resume):
            raise FileNotFoundError(f"no checkpoint found at '{resume}'")
        checkpoint = torch.load(resume, map_location=device)

        if checkpoint["model_params"]["robust"] != robust:
            raise ValueError(f"robustness of checkpoint '{resume}' is not {robust}")

        chkpt_task_dict = checkpoint["model_params"]["task_dict"]
        if chkpt_task_dict != task_dict:
            raise ValueError(
                f"task_dict {chkpt_task_dict} of checkpoint '{resume}' does not match provided "
                f"task_dict {task_dict}"
            )

        model = model_class(**checkpoint["model_params"])
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        normalizer_dict: dict[str, Normalizer] = {}
        for task, state_dict in checkpoint["normalizer_dict"].items():
            if state_dict is not None:
                normalizer_dict[task] = Normalizer.from_state_dict(state_dict)
            else:
                normalizer_dict[task] = None

        y_test, output, *ids = model.predict(generator=test_generator)

        for pred, target, (name, task) in zip(output, y_test, model.task_dict.items()):
            if task == "regression":
                if model.robust:
                    mean, log_std = pred.chunk(2, dim=1)
                    pred = normalizer_dict[name].denorm(mean.data.cpu())
                    ale_std = torch.exp(log_std).data.cpu() * normalizer_dict[name].std
                    results_dict[name]["ale"][j, :] = ale_std.view(-1).numpy()  # type: ignore
                else:
                    pred = normalizer_dict[name].denorm(pred.data.cpu())

                results_dict[name]["pred"][j, :] = pred.view(-1).numpy()  # type: ignore

            elif task == "classification":
                if model.robust:
                    mean, log_std = pred.chunk(2, dim=1)
                    logits = (
                        sampled_softmax(mean, log_std, samples=10).data.cpu().numpy()
                    )
                    pre_logits = mean.data.cpu().numpy()
                    pre_logits_std = torch.exp(log_std).data.cpu().numpy()
                    results_dict[name]["pre-logits_ale"].append(pre_logits_std)  # type: ignore
                else:
                    pre_logits = pred.data.cpu().numpy()
                    logits = softmax(pre_logits, axis=1)

                results_dict[name]["pre-logits"].append(pre_logits)  # type: ignore
                results_dict[name]["logits"].append(logits)  # type: ignore

            results_dict[name]["target"] = target

    # TODO cleaner way to get identifier names
    if save_results:
        save_results_dict(
            dict(zip(test_generator.dataset.dataset.identifiers, ids)),
            results_dict,
            model_name,
        )

    if print_results:
        for name, task in task_dict.items():
            print(f"\nTask: '{name}' on test set")
            if task == "regression":
                print_metrics_regression(**results_dict[name])  # type: ignore
            elif task == "classification":
                print_metrics_classification(**results_dict[name])  # type: ignore

    return results_dict


def print_metrics_regression(target: Tensor, pred: Tensor, **kwargs) -> None:
    """Print out metrics for a regression task.

    Args:
        target (ndarray(n_test)): targets for regression task
        pred (ndarray(n_ensemble, n_test)): model predictions
        kwargs: unused entries from the results dictionary
    """
    ensemble_folds = pred.shape[0]
    res = pred - target
    mae = np.mean(np.abs(res), axis=1)
    mse = np.mean(np.square(res), axis=1)
    rmse = np.sqrt(mse)
    r2 = r2_score(
        np.repeat(target[:, np.newaxis], ensemble_folds, axis=1),
        pred.T,
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
        y_ens = np.mean(pred, axis=0)

        mae_ens = np.abs(target - y_ens).mean()
        mse_ens = np.square(target - y_ens).mean()
        rmse_ens = np.sqrt(mse_ens)

        r2_ens = r2_score(target, y_ens)

        print("\nEnsemble Performance Metrics:")
        print(f"R2 Score : {r2_ens:.4f} ")
        print(f"MAE  : {mae_ens:.4f}")
        print(f"RMSE : {rmse_ens:.4f}")


def print_metrics_classification(
    target: LongTensor,
    logits: Tensor,
    average: Literal["micro", "macro", "samples", "weighted"] = "micro",
    **kwargs,
) -> None:
    """Print out metrics for a classification task.

    TODO make less janky, first index is for ensembles, second data, third classes.
    always calculate metrics in the multi-class setting. How to convert binary labels
    to multi-task automatically?

    Args:
        target (ndarray(n_test)): categorical encoding of the tasks
        logits (list[n_ens * ndarray(n_targets, n_test)]): logits predicted by the model
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
    target_ohe[np.arange(target.size), target] = 1

    for j, y_logit in enumerate(logits):

        y_pred = np.argmax(y_logit, axis=1)

        acc[j] = accuracy_score(target, y_pred)
        roc_auc[j] = roc_auc_score(target_ohe, y_logit, average=average)
        precision[j], recall[j], fscore[j], _ = precision_recall_fscore_support(
            target, y_pred, average=average
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

        ens_acc = accuracy_score(target, y_pred)
        ens_roc_auc = roc_auc_score(target_ohe, ens_logits, average=average)
        ens_prec, ens_recall, ens_fscore, _ = precision_recall_fscore_support(
            target, y_pred, average=average
        )

        print("\nEnsemble Performance Metrics:")
        print(f"Accuracy : {ens_acc:.4f} ")
        print(f"ROC-AUC  : {ens_roc_auc:.4f}")
        print(f"Weighted Precision : {ens_prec:.4f}")
        print(f"Weighted Recall    : {ens_recall:.4f}")
        print(f"Weighted F-score   : {ens_fscore:.4f}")


def save_results_dict(
    ids: dict[str, list[str | int]], results_dict: dict[str, Any], model_name: str
) -> None:
    """Save the results to a file after model evaluation.

    Args:
        ids (dict[str, list[str  |  int]]): ): Each key is the name of an identifier (e.g.
            material ID, composition, ...) and its value a list of IDs.
        results_dict (dict[str, Any]): ): nested dictionary of results {name: {col: data}}
        model_name (str): ): The name given the model via the --model-name flag.
    """
    results = {}

    for target_name in results_dict:
        for col, data in results_dict[target_name].items():

            # NOTE we save pre_logits rather than logits due to fact
            # that with the heteroskedastic setup we want to be able to
            # sample from the Gaussian distributed pre_logits we parameterise.
            if "pre-logits" in col:
                for n_ens, y_pre_logit in enumerate(data):
                    results.update(
                        {
                            f"{target_name}_{col}_c{lab}_n{n_ens}": val.ravel()
                            for lab, val in enumerate(y_pre_logit.T)
                        }
                    )

            elif "pred" in col:
                preds = {
                    f"{target_name}_{col}_n{n_ens}": val.ravel()
                    for (n_ens, val) in enumerate(data)
                }
                results.update(preds)

            elif "ale" in col:  # elif so that pre-logit-ale doesn't trigger
                results.update(
                    {
                        f"{target_name}_{col}_n{n_ens}": val.ravel()
                        for (n_ens, val) in enumerate(data)
                    }
                )

            elif col == "target":
                results.update({f"{target_name}_target": data})

    df = pd.DataFrame({**ids, **results})

    file_name = model_name.replace("/", "_")

    os.makedirs("results", exist_ok=True)

    csv_path = f"results/{file_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved model predictions to '{csv_path}'")


def softmax(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute the softmax of an array along an axis.

    Args:
        arr (np.ndarray): Arbitrary dimensional array.
        axis (int, optional): Dimension over which to take softmax. Defaults to -1 (last).

    Returns:
        np.ndarray: Same dimension as input array, but specified axis reduced
            to singleton.
    """
    exp = np.exp(arr)
    return exp / exp.sum(axis=axis, keepdims=True)


def one_hot(targets: np.ndarray, n_classes: int = None) -> np.ndarray:
    """Get a one-hot encoded version of an array of class labels.

    Args:
        targets (np.ndarray): 1d array of integer class labels.
        n_classes (int, optional): Number of classes. Defaults to np.max(targets) + 1.

    Returns:
        np.ndarray: 2d array of 1-hot encoded class labels.
    """
    if n_classes is None:
        n_classes = np.max(targets) + 1
    return np.eye(n_classes)[targets]
