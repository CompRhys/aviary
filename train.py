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
from sklearn.model_selection import train_test_split as split

from scipy.special import softmax

from roost.model import Roost, ResidualNetwork
from roost.data import input_parser, CompositionData, collate_batch
from roost.utils import Normalizer, sampled_softmax, RobustL1, RobustL2


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
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    **kwargs,
):
    assert evaluate or train, ("No task given - Set at least one of "
        "'train' or 'evaluate' kwargs as True")
    assert task in ["regression", "classification"], ("Only "
        "'regression' or 'classification' allowed for 'task'")

    if test_path or (not evaluate):
        test_size = 0.0

    if not (test_path and val_path):
        assert test_size + val_size < 1.0, (f"'test_size'({test_size}) "
            f"plus 'val_size'({val_size}) must be less than 1")

    if ensemble > 1 and (fine_tune or transfer):
        raise NotImplementedError(
            "If training an ensemble with fine tuning or transfering"
            " options the models must be trained one by one using the"
            " run-id flag."
        )

    assert not (fine_tune and transfer), ("Cannot fine-tune and"
        " transfer checkpoint(s) at the same time.")

    dataset = CompositionData(data_path=data_path, fea_path=fea_path, task=task)
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
                # NOTE that when using this option care must be taken not to
                # peak at the test-set. The only valid model to use is the one
                # obtained after the final epoch where the epoch count is
                # decided in advance of the experiment.
                val_set = test_set
            elif val_size == 0.0:
                val_set = None
            else:
                print(f"using {val_size} of training set as validation set")
                train_idx, val_idx = split(
                    train_idx, random_state=seed, test_size=val_size / (1 - test_size),
                )
                val_set = torch.utils.data.Subset(dataset, val_idx)

        train_set = torch.utils.data.Subset(dataset, train_idx[0::sample])

    data_params = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": collate_batch,
    }

    setup_params = {
        "loss": loss,
        "optim": optim,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "device": device,
    }

    restart_params = {
        "resume": resume,
        "fine_tune": fine_tune,
        "transfer": transfer,
    }

    model_params = {
        "task": task,
        "robust": robust,
        "n_targets": n_targets,
        "elem_emb_len": elem_emb_len,
        "elem_fea_len": elem_fea_len,
        "n_graph": n_graph,
        "elem_heads": 3,
        "elem_gate": [256],
        "elem_msg": [256],
        "cry_heads": 3,
        "cry_gate": [256],
        "cry_msg": [256],
        "out_hidden": [1024, 512, 256, 128, 64],
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
            setup_params=setup_params,
            restart_params=restart_params,
            model_params=model_params,
        )

    if evaluate:

        data_reset = {
            "batch_size": 16 * batch_size,  # faster model inference
            "shuffle": False,  # need fixed data order due to ensembling
        }
        data_params.update(data_reset)

        if task == "regression":
            results_regression(
                model_name=model_name,
                run_id=run_id,
                ensemble_folds=ensemble,
                test_set=test_set,
                data_params=data_params,
                robust=robust,
                device=device,
                eval_type="checkpoint",
            )
        elif task == "classification":
            results_classification(
                model_name=model_name,
                run_id=run_id,
                ensemble_folds=ensemble,
                test_set=test_set,
                data_params=data_params,
                robust=robust,
                device=device,
                eval_type="checkpoint",
            )


def init_model(
    model_name,
    run_id,
    task,
    robust,
    loss,
    optim,
    learning_rate,
    weight_decay,
    momentum,
    device,
    n_targets,
    elem_emb_len,
    elem_fea_len,
    n_graph,
    elem_heads,
    elem_gate,
    elem_msg,
    cry_heads,
    cry_gate,
    cry_msg,
    out_hidden,
    resume=None,
    fine_tune=None,
    transfer=None,
):

    if fine_tune is not None:
        print(f"Use material_nn and output_nn from '{fine_tune}' as a starting point")
        checkpoint = torch.load(fine_tune, map_location=device)
        model = Roost(**checkpoint["model_params"], device=device,)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])

        assert model.model_params["robust"] == robust, ("cannot fine-tune "
            "between tasks with different numebers of outputs - use transfer "
            "option instead")
        assert model.model_params["n_targets"] == n_targets, ("cannot fine-tune "
            "between tasks with different numebers of outputs - use transfer "
            "option instead")

    elif transfer is not None:
        print(f"Use material_nn from '{transfer}' as a starting point and "
            "train the output_nn from scratch")
        checkpoint = torch.load(transfer, map_location=device)
        model = Roost(**checkpoint["model_params"], device=device,)
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
            input_dim=elem_fea_len, hidden_layer_dims=out_hidden, output_dim=output_dim,
        )

    else:
        model = Roost(
            n_targets=n_targets,
            elem_emb_len=elem_emb_len,
            elem_fea_len=elem_fea_len,
            n_graph=n_graph,
            elem_heads=elem_heads,
            elem_gate=elem_gate,
            elem_msg=elem_msg,
            cry_heads=cry_heads,
            cry_gate=cry_gate,
            cry_msg=cry_msg,
            out_hidden=out_hidden,
            task=task,
            robust=robust,
            device=device,
        )

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

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [])

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

    if args.resume:
        # TODO work out how to ensure that we are using the same optimizer
        # when resuming such that the state dictionaries do not clash.
        resume = f"models/{model_name}/checkpoint-r{run_id}.pth.tar"
        print(f"Resuming training from '{resume}'")
        checkpoint = torch.load(resume, map_location=device)

        model = Roost(**checkpoint["model_params"], device=device,)
        model.to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.epoch = checkpoint["epoch"]
        model.best_val_score = checkpoint["best_val_score"]

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
            model_name=model_name,
            run_id=j,
            **setup_params,
            **restart_params,
            **model_params,
        )

        if model.task == "regression":
            sample_target = torch.Tensor(
                train_set.dataset.df.iloc[train_set.indices, 2].values
            )
            normalizer.fit(sample_target)

        if log:
            writer = SummaryWriter(
                log_dir=(f"runs/{model_name}-r{j}_" "{date:%d-%m-%Y_%H-%M-%S}").format(
                    date=datetime.datetime.now()
                )
            )
        else:
            writer = None

        if (val_set is not None) and (model.best_val_score is None):
            with torch.no_grad():
                _, v_metrics = model.evaluate(
                    generator=val_generator,
                    criterion=criterion,
                    optimizer=None,
                    normalizer=normalizer,
                    action="val",
                )
                if model.task == "regression":
                    val_score = v_metrics["MAE"]
                elif model.task == "classification":
                    val_score = v_metrics["Acc"]
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

        model = Roost(**checkpoint["model_params"], device=device,)
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

        model = Roost(**checkpoint["model_params"], device=device,)
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


if __name__ == "__main__":
    args = input_parser()

    print(f"The model will run on the {args.device} device")

    main(**vars(args))
