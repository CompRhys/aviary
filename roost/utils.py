import os
import datetime

import numpy as np
import pandas as pd

import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, NLLLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    r2_score,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
)

from scipy.special import softmax

from roost.core import Normalizer, sampled_softmax, RobustL1, RobustL2
from roost.segments import ResidualNetwork


def init_model(
    model_class,
    model_name,
    model_params,
    run_id,
    loss,
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

    task = model_params["task"]
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
            "between tasks with different numebers of outputs - use transfer "
            "option instead"
        )
        assert model.model_params["n_targets"] == n_targets, (
            "cannot fine-tune "
            "between tasks with different numebers of outputs - use transfer "
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

        model.task = task
        model.model_params["task"] = task
        model.robust = robust
        model.model_params["robust"] = robust
        model.model_params["n_targets"] = n_targets

        # # NOTE currently if you use a model as a feature extractor and then
        # # resume for a checkpoint of that model the material_nn unfreezes.
        # # This is potentially not the behaviour a user might expect.
        # for p in model.material_nn.parameters():
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
        resume = f"models/{model_name}/checkpoint-r{run_id}.pth.tar"
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
        optimizer,
        milestones=milestones,
        gamma=gamma
    )

    # Select Task and Loss Function
    if task == "classification":
        normalizer = None
        if robust:
            criterion = NLLLoss()
        else:
            criterion = CrossEntropyLoss()

    elif task == "regression":
        normalizer = Normalizer()
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
                raise NameError("Only L1 or L2 losses are allowed for regression tasks")

    if resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        normalizer.load_state_dict(checkpoint["normalizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Number of Trainable Parameters: {}".format(num_param))

    # TODO parallelise the code over multiple GPUs. Currently DataParallel
    # crashes as subsets of the batch have different sizes due to the use of
    # lists of lists rather the zero-padding.
    # if (torch.cuda.device_count() > 1) and (device==torch.device("cuda")):
    #     print("The model will use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    model.to(device)

    return model, criterion, optimizer, scheduler, normalizer


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
        #  by specifiying the run-id arg.
        if ensemble_folds == 1:
            j = run_id

        model, criterion, optimizer, scheduler, normalizer = init_model(
            model_class=model_class,
            model_name=model_name,
            model_params=model_params,
            run_id=j,
            **setup_params,
            **restart_params,
        )

        if model.task == "regression":
            sample_target = torch.Tensor(
                train_set.dataset.df.iloc[train_set.indices, 2].values
            )
            if not restart_params["resume"]:
                normalizer.fit(sample_target)
            print(f"Dummy MAE: {torch.mean(torch.abs(sample_target-normalizer.mean)):.4f}")

        if log:
            writer = SummaryWriter(
                log_dir=(f"runs/{model_name}-r{j}_" "{date:%d-%m-%Y_%H-%M-%S}").format(
                    date=datetime.datetime.now()
                )
            )
        else:
            writer = None

        if (val_set is not None) and (model.best_val_score is None):
            print("Getting Validation Baseline")
            with torch.no_grad():
                _, v_metrics = model.evaluate(
                    generator=val_generator,
                    criterion=criterion,
                    optimizer=None,
                    normalizer=normalizer,
                    action="val",
                    verbose=True,
                )
                if model.task == "regression":
                    val_score = v_metrics["MAE"]
                    print(f"Validation Baseline: MAE {val_score:.3f}\n")
                elif model.task == "classification":
                    val_score = v_metrics["Acc"]
                    print(f"Validation Baseline: Acc {val_score:.3f}\n")
                model.best_val_score = val_score

        model.fit(
            train_generator=train_generator,
            val_generator=val_generator,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            criterion=criterion,
            normalizer=normalizer,
            model_name=model_name,
            run_id=j,
            writer=writer,
        )


def results_regression(
    model_class,
    model_name,
    run_id,
    ensemble_folds,
    test_set,
    data_params,
    robust,
    device,
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

    y_ensemble = np.zeros((ensemble_folds, len(test_set)))
    if robust:
        y_ale = np.zeros((ensemble_folds, len(test_set)))

    for j in range(ensemble_folds):

        if ensemble_folds == 1:
            resume = f"models/{model_name}/{eval_type}-r{run_id}.pth.tar"
            print("Evaluating Model")
        else:
            resume = f"models/{model_name}/{eval_type}-r{j}.pth.tar"
            print("Evaluating Model {}/{}".format(j + 1, ensemble_folds))

        assert os.path.isfile(resume), f"no checkpoint found at '{resume}'"
        checkpoint = torch.load(resume, map_location=device)
        checkpoint["model_params"]["robust"]
        assert (
            checkpoint["model_params"]["robust"] == robust
        ), f"robustness of checkpoint '{resume}' is not {robust}"

        model = model_class(**checkpoint["model_params"], device=device,)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        normalizer = Normalizer()
        normalizer.load_state_dict(checkpoint["normalizer"])

        with torch.no_grad():
            idx, comp, y_test, output = model.predict(generator=test_generator,)

        if robust:
            mean, log_std = output.chunk(2, dim=1)
            pred = normalizer.denorm(mean.data.cpu())
            ale_std = torch.exp(log_std).data.cpu() * normalizer.std
            y_ale[j, :] = ale_std.view(-1).numpy()
        else:
            pred = normalizer.denorm(output.data.cpu())

        y_ensemble[j, :] = pred.view(-1).numpy()

    res = y_ensemble - y_test
    mae = np.mean(np.abs(res), axis=1)
    mse = np.mean(np.square(res), axis=1)
    rmse = np.sqrt(mse)
    r2 = r2_score(
        np.repeat(y_test[:, np.newaxis], ensemble_folds, axis=1),
        y_ensemble.T,
        multioutput="raw_values",
    )

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
        rmse_ens = np.sqrt(mse_ens)

        r2_ens = r2_score(y_test, y_ens)

        print("\nEnsemble Performance Metrics:")
        print(f"R2 Score : {r2_ens:.4f} ")
        print(f"MAE  : {mae_ens:.4f}")
        print(f"RMSE : {rmse_ens:.4f}")

    core = {"id": idx, "composition": comp, "target": y_test}
    results = {f"pred_{n_ens}": val for (n_ens, val) in enumerate(y_ensemble)}
    if model.robust:
        ale = {f"ale_{n_ens}": val for (n_ens, val) in enumerate(y_ale)}
        results.update(ale)

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
    model_class,
    model_name,
    run_id,
    ensemble_folds,
    test_set,
    data_params,
    robust,
    device,
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

    y_pre_logits = []
    y_logits = []
    if robust:
        y_pre_ale = []

    acc = np.zeros((ensemble_folds))
    roc_auc = np.zeros((ensemble_folds))
    precision = np.zeros((ensemble_folds))
    recall = np.zeros((ensemble_folds))
    fscore = np.zeros((ensemble_folds))

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

        model = model_class(**checkpoint["model_params"], device=device,)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        with torch.no_grad():
            idx, comp, y_test, output = model.predict(generator=test_generator)

        if model.robust:
            mean, log_std = output.chunk(2, dim=1)
            logits = sampled_softmax(mean, log_std, samples=10).data.cpu().numpy()
            pre_logits = mean.data.cpu().numpy()
            pre_logits_std = torch.exp(log_std).data.cpu().numpy()
            y_pre_ale.append(pre_logits_std)
        else:
            pre_logits = output.data.cpu().numpy()

        logits = softmax(pre_logits, axis=1)

        y_pre_logits.append(pre_logits)
        y_logits.append(logits)

        y_test_ohe = np.zeros_like(pre_logits)
        y_test_ohe[np.arange(y_test.size), y_test] = 1

        acc[j] = accuracy_score(y_test, np.argmax(logits, axis=1))
        roc_auc[j] = roc_auc_score(y_test_ohe, logits)
        precision[j], recall[j], fscore[j] = precision_recall_fscore_support(
            y_test, np.argmax(logits, axis=1), average="weighted"
        )[:3]

    if ensemble_folds == 1:
        print("\nModel Performance Metrics:")
        print("Accuracy : {:.4f} ".format(acc[0]))
        print("ROC-AUC  : {:.4f}".format(roc_auc[0]))
        print("Weighted Precision : {:.4f}".format(precision[0]))
        print("Weighted Recall    : {:.4f}".format(recall[0]))
        print("Weighted F-score   : {:.4f}".format(fscore[0]))
    else:
        acc_avg = np.mean(acc)
        acc_std = np.std(acc)

        roc_auc_avg = np.mean(roc_auc)
        roc_auc_std = np.std(roc_auc)

        precision_avg = np.mean(precision)
        precision_std = np.std(precision)

        recall_avg = np.mean(recall)
        recall_std = np.std(recall)

        fscore_avg = np.mean(fscore)
        fscore_std = np.std(fscore)

        print("\nModel Performance Metrics:")
        print(f"Accuracy : {acc_avg:.4f} +/- {acc_std:.4f}")
        print(f"ROC-AUC  : {roc_auc_avg:.4f} +/- {roc_auc_std:.4f}")
        print(f"Weighted Precision : {precision_avg:.4f} +/- {precision_std:.4f}")
        print(f"Weighted Recall    : {recall_avg:.4f} +/- {recall_std:.4f}")
        print(f"Weighted F-score   : {fscore_avg:.4f} +/- {fscore_std:.4f}")

        # calculate metrics and errors with associated errors for ensembles
        ens_logits = np.mean(y_logits, axis=0)

        y_test_ohe = np.zeros_like(ens_logits)
        y_test_ohe[np.arange(y_test.size), y_test] = 1

        ens_acc = accuracy_score(y_test, np.argmax(ens_logits, axis=1))
        ens_roc_auc = roc_auc_score(y_test_ohe, ens_logits)
        ens_precision, ens_recall, ens_fscore = precision_recall_fscore_support(
            y_test, np.argmax(ens_logits, axis=1), average="weighted"
        )[:3]

        print("\nEnsemble Performance Metrics:")
        print(f"Accuracy : {ens_acc:.4f} ")
        print(f"ROC-AUC  : {ens_roc_auc:.4f}")
        print(f"Weighted Precision : {ens_precision:.4f}")
        print(f"Weighted Recall    : {ens_recall:.4f}")
        print(f"Weighted F-score   : {ens_fscore:.4f}")

    # NOTE we save pre_logits rather than logits due to fact that with the
    # hetroskedastic setup we want to be able to sample from the gaussian
    # distributed pre_logits we parameterise.
    core = {"id": idx, "composition": comp, "target": y_test}

    results = {}
    for n_ens, y_pre_logit in enumerate(y_pre_logits):
        pred_dict = {
            f"class-{lab}-pred_{n_ens}": val for lab, val in enumerate(y_pre_logit.T)
        }
        results.update(pred_dict)
        if model.robust:
            ale_dict = {
                f"class-{lab}-ale_{n_ens}": val
                for lab, val in enumerate(y_pre_ale[n_ens].T)
            }
            results.update(ale_dict)

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
