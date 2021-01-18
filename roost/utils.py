import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, NLLLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from roost.core import Normalizer, RobustL1Loss, RobustL2Loss, sampled_softmax
from roost.segments import ResidualNetwork


def init_model(
    model_class,
    model_name,
    model_params,
    run_id,
    optim,
    learning_rate,
    weight_decay,
    momentum,
    device,
    milestones=[],
    gamma=0.3,
    resume=None,
    fine_tune=None,
    transfer=None,
):

    robust = model_params["robust"]
    n_targets = model_params["n_targets"]

    if fine_tune is not None:
        print(f"Use material_nn and output_nn from '{fine_tune}' as a starting point")
        checkpoint = torch.load(fine_tune, map_location=device)
        model = model_class(**checkpoint["model_params"], device=device,)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        assert model.model_params["robust"] == robust, (
            "cannot fine-tune "
            "between tasks with different numbers of outputs - use transfer "
            "option instead"
        )
        assert model.model_params["n_targets"] == n_targets, (
            "cannot fine-tune "
            "between tasks with different numbers of outputs - use transfer "
            "option instead"
        )

    elif transfer is not None:
        print(
            f"Use material_nn from '{transfer}' as a starting point and "
            "train the output_nn from scratch"
        )
        checkpoint = torch.load(transfer, map_location=device)
        model = model_class(**checkpoint["model_params"], device=device,)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        model.robust = robust
        model.model_params["robust"] = robust
        model.model_params["n_targets"] = n_targets

        # # NOTE currently if you use a model as a feature extractor and then
        # # resume for a checkpoint of that model the material_nn unfreezes.
        # # This is potentially not the behaviour a user might expect.
        # for p in model.material_nn.parameters():
        #     p.requires_grad = False
        # for p in model.trunk_nn.parameters():
        #     p.requires_grad = False

        if robust:
            output_dim = 2 * n_targets
        else:
            output_dim = n_targets

        model.output_nn = ResidualNetwork(
            input_dim=model_params["elem_fea_len"],
            hidden_layer_dims=model_params["out_hidden"],
            output_dim=output_dim,
        )

    elif resume:
        # TODO work out how to ensure that we are using the same optimizer
        # when resuming such that the state dictionaries do not clash.
        print(f"Resuming training from '{resume}'")
        checkpoint = torch.load(resume, map_location=device)

        model = model_class(**checkpoint["model_params"], device=device,)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.epoch = checkpoint["epoch"]
        model.best_val_score = checkpoint["best_val_score"]

    else:
        model = model_class(device=device, **model_params)

        model.to(device)

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
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optim == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise NameError("Only SGD, Adam or AdamW are allowed as --optim")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )

    if resume:
        # NOTE the user could change the optimizer when resuming creating a bug
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    print(f"Total Number of Trainable Parameters: {model.num_params}")

    # TODO parallelise the code over multiple GPUs. Currently DataParallel
    # crashes as subsets of the batch have different sizes due to the use of
    # lists of lists rather the zero-padding.
    # if (torch.cuda.device_count() > 1) and (device==torch.device("cuda")):
    #     print("The model will use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model.to(device)

    return model, optimizer, scheduler


def init_losses(task_dict, loss_dict, robust=False):

    criterion_dict = {}
    for name, task in task_dict.items():
        # Select Task and Loss Function
        if task == "classification":
            if loss_dict[name] != "CSE":
                raise NameError(
                    "Only CSE loss allowed for classification tasks"
                )

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
                    raise NameError("Only L1 or L2 losses are allowed for regression tasks")

    return criterion_dict


def init_normalizers(task_dict, device, resume=False):
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        normalizer_dict = {}
        for task, state_dict in checkpoint["normalizer_dict"].items():
            normalizer_dict[task] = Normalizer.from_state_dict(state_dict)

        return normalizer_dict

    normalizer_dict = {}
    for target, task in task_dict.items():
        # Select Task and Loss Function
        if task == "classification":
            normalizer_dict[target] = None
        elif task == "regression":
            normalizer_dict[target] = Normalizer()

    return normalizer_dict


def train_ensemble(
    model_class,
    model_name,
    run_id,
    ensemble_folds,
    epochs,
    train_set,
    val_set,
    log,
    data_params,
    setup_params,
    restart_params,
    model_params,
    loss_dict,
    patience=None,
):
    """
    Train multiple models
    """

    train_generator = DataLoader(train_set, **data_params)

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

        model, optimizer, scheduler = init_model(
            model_class=model_class,
            model_name=model_name,
            model_params=model_params,
            run_id=j,
            **setup_params,
            **restart_params,
        )

        criterion_dict = init_losses(
            model.task_dict,
            loss_dict,
            model_params["robust"]
        )
        normalizer_dict = init_normalizers(
            model.task_dict,
            setup_params["device"],
            restart_params["resume"]
        )

        for target, normalizer in normalizer_dict.items():
            if normalizer is not None:
                sample_target = torch.Tensor(
                    train_set.dataset.df[target].iloc[train_set.indices].values
                )
                if not restart_params["resume"]:
                    normalizer.fit(sample_target)
                print(
                    f"Dummy MAE: {torch.mean(torch.abs(sample_target-normalizer.mean)):.4f}"
                )

        if log:
            writer = SummaryWriter(
                log_dir=(f"runs/{model_name}-r{j}_{datetime.now():%d-%m-%Y_%H-%M-%S}")
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
                    verbose=True,
                )

                val_score = {}

                for name, task in model.task_dict.items():
                    if task == "regression":
                        val_score[name] = v_metrics[name]["MAE"]
                        print(f"Validation Baseline - {name}: MAE {val_score[name]:.3f}")
                    elif task == "classification":
                        val_score[name] = v_metrics[name]["Acc"]
                        print(f"Validation Baseline - {name}: Acc {val_score[name]:.3f}")
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


@torch.no_grad()
def results_multitask(
    model_class,
    model_name,
    run_id,
    ensemble_folds,
    test_set,
    data_params,
    robust,
    task_dict,
    device,
    eval_type="checkpoint",
    print_results=True,
    save_results=True,
):
    """
    take an ensemble of models and evaluate their performance on the test set
    """

    assert print_results or save_results, (
        "Evaluating Model pointless if both 'print_results' and "
        "'save_results' are False."
    )

    print(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        "------------Evaluate model on Test Set------------\n"
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
    )

    test_generator = DataLoader(test_set, **data_params)

    results_dict = {n: {} for n in task_dict}
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
            resume = f"models/{model_name}/{eval_type}-r{run_id}.pth.tar"
            print("Evaluating Model")
        else:
            resume = f"models/{model_name}/{eval_type}-r{j}.pth.tar"
            print("Evaluating Model {}/{}".format(j + 1, ensemble_folds))

        assert os.path.isfile(resume), f"no checkpoint found at '{resume}'"
        checkpoint = torch.load(resume, map_location=device)

        assert (
            checkpoint["model_params"]["robust"] == robust
        ), f"robustness of checkpoint '{resume}' is not {robust}"

        assert (
            checkpoint["model_params"]["task_dict"] == task_dict
        ), f"task_dict of checkpoint '{resume}' does not match current task_dict"

        model = model_class(**checkpoint["model_params"], device=device)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        normalizer_dict = {}
        for task, state_dict in checkpoint["normalizer_dict"].items():
            normalizer_dict[task] = Normalizer.from_state_dict(state_dict)

        idx, comp, y_test, output = model.predict(generator=test_generator)

        # TODO should output also be a dictionary?

        for pred, target, (name, task) in zip(output, y_test, model.task_dict.items()):
            if task == "regression":
                if model.robust:
                    mean, log_std = pred.chunk(2, dim=1)
                    pred = normalizer_dict[name].denorm(mean.data.cpu())
                    ale_std = torch.exp(log_std).data.cpu() * normalizer_dict[name].std
                    results_dict[name]["ale"][j, :] = ale_std.view(-1).numpy()
                else:
                    pred = normalizer_dict[name].denorm(pred.data.cpu())

                results_dict[name]["pred"][j, :] = pred.view(-1).numpy()

            elif task == "classification":
                if model.robust:
                    mean, log_std = pred.chunk(2, dim=1)
                    logits = sampled_softmax(mean, log_std, samples=10).data.cpu().numpy()
                    pre_logits = mean.data.cpu().numpy()
                    pre_logits_std = torch.exp(log_std).data.cpu().numpy()
                    results_dict[name]["pre-logits-ale"].append(pre_logits_std)
                else:
                    pre_logits = pred.data.cpu().numpy()

                logits = softmax(pre_logits, axis=1)

                results_dict[name]["pre-logits"].append(pre_logits)
                results_dict[name]["logits"].append(logits)

            results_dict[name]["target"] = target

    if print_results:
        for name, task in task_dict.items():
            print(f"\nTask: '{name}' on Test Set")
            if task == "regression":
                print_metrics_regression(**results_dict[name])
            elif task == "classification":
                print_metrics_classification(**results_dict[name])

    if save_results:
        save_results_dict(idx, comp, results_dict, model_name)

    return results_dict


def print_metrics_regression(target, pred, **kwargs):
    """print out metrics for a regression task

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
    mae_std = np.std(mae)/np.sqrt(mae.shape[0])

    rmse_avg = np.mean(rmse)
    rmse_std = np.std(rmse)/np.sqrt(rmse.shape[0])

    if ensemble_folds == 1:
        print("Model Performance Metrics:")
        print("R2 Score: {:.4f} ".format(r2_avg))
        print("MAE: {:.4f}".format(mae_avg))
        print("RMSE: {:.4f}".format(rmse_avg))
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


def print_metrics_classification(target, logits, **kwargs):
    """print out metrics for a classification task

    Args:
        target (ndarray(n_test)): categorical encoding of the tasks
        logits (ndarray(n_test, n_targets)): logits predicted by the model
        kwargs: unused entries from the results dictionary
    """
    acc = np.zeros(len(logits))
    roc_auc = np.zeros(len(logits))
    precision = np.zeros(len(logits))
    recall = np.zeros(len(logits))
    fscore = np.zeros(len(logits))

    for y_logit in logits:

        target_ohe = np.zeros_like(y_logit)
        target_ohe[np.arange(target.size), target] = 1

        acc[j] = accuracy_score(target, np.argmax(y_logit, axis=1))
        roc_auc[j] = roc_auc_score(target_ohe, y_logit)
        precision[j], recall[j], fscore[j] = precision_recall_fscore_support(
            target, np.argmax(logits[j], axis=1), average="weighted"
        )[:3]

    if len(logits) == 1:
        print("\nModel Performance Metrics:")
        print("Accuracy : {:.4f} ".format(acc[0]))
        print("ROC-AUC  : {:.4f}".format(roc_auc[0]))
        print("Weighted Precision : {:.4f}".format(precision[0]))
        print("Weighted Recall    : {:.4f}".format(recall[0]))
        print("Weighted F-score   : {:.4f}".format(fscore[0]))
    else:
        acc_avg = np.mean(acc)
        acc_std = np.std(acc)/np.sqrt(acc.shape[0])

        roc_auc_avg = np.mean(roc_auc)
        roc_auc_std = np.std(roc_auc)/np.sqrt(roc_auc.shape[0])

        prec_avg = np.mean(precision)
        prec_std = np.std(precision)/np.sqrt(precision.shape[0])

        recall_avg = np.mean(recall)
        recall_std = np.std(recall)/np.sqrt(recall.shape[0])

        fscore_avg = np.mean(fscore)
        fscore_std = np.std(fscore)/np.sqrt(fscore.shape[0])

        print("\nModel Performance Metrics:")
        print(f"Accuracy : {acc_avg:.4f} +/- {acc_std:.4f}")
        print(f"ROC-AUC  : {roc_auc_avg:.4f} +/- {roc_auc_std:.4f}")
        print(f"Weighted Precision : {prec_avg:.4f} +/- {prec_std:.4f}")
        print(f"Weighted Recall    : {recall_avg:.4f} +/- {recall_std:.4f}")
        print(f"Weighted F-score   : {fscore_avg:.4f} +/- {fscore_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        ens_logits = np.mean(logits, axis=0)

        target_ohe = np.zeros_like(ens_logits)
        target_ohe[np.arange(target.size), target] = 1

        ens_acc = accuracy_score(target, np.argmax(ens_logits, axis=1))
        ens_roc_auc = roc_auc_score(target_ohe, ens_logits)
        ens_prec, ens_recall, ens_fscore = precision_recall_fscore_support(
            target, np.argmax(ens_logits, axis=1), average="weighted"
        )[:3]

        print("\nEnsemble Performance Metrics:")
        print(f"Accuracy : {ens_acc:.4f} ")
        print(f"ROC-AUC  : {ens_roc_auc:.4f}")
        print(f"Weighted Precision : {ens_prec:.4f}")
        print(f"Weighted Recall    : {ens_recall:.4f}")
        print(f"Weighted F-score   : {ens_fscore:.4f}")


def save_results_dict(idx, comp, results_dict, model_name):
    """save the results to a file after model evaluation

    Args:
        idx ([str]): list of unique identifiers
        comp ([str]): list of compositions
        results_dict ({name: {col: data}}): nested dictionary of results
        model_name (str): [description]
    """
    core = {"id": idx, "composition": comp}
    results = {}

    for name in results_dict:
        for col, data in results_dict[name].items():

            # NOTE we save pre_logits rather than logits due to fact
            # that with the hetroskedastic setup we want to be able to
            # sample from the gaussian distributed pre_logits we parameterise.
            if "pre-logits" in col:
                for n_ens, y_pre_logit in enumerate(data):
                    results.update({
                        f"{name}_{col}_c{lab}_n{n_ens}": val.ravel()
                        for lab, val in enumerate(y_pre_logit.T)
                    })

            elif "pred" in col:
                preds = {
                    f"{name}_{col}_n{n_ens}": val.ravel()
                    for (n_ens, val) in enumerate(data)
                }
                results.update(preds)

            elif "ale" in col:  # elif so that pre-logit-ale doesn't trigger
                results.update({
                    f"{name}_{col}_n{n_ens}": val.ravel()
                    for (n_ens, val) in enumerate(data)
                })

            elif col == "target":
                results.update({f"{name}_{col}": data})

    df = pd.DataFrame({**core, **results})

    df.to_csv(
        index=False,
        path_or_buf=(f"results/multi_results_{model_name}.csv")
    )
