from __future__ import annotations

import os
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import torch

from aviary import ROOT
from aviary.core import Normalizer, TaskType
from aviary.losses import RobustL1Loss
from aviary.utils import get_metrics
from aviary.wrenformer.data import df_to_in_mem_dataloader
from aviary.wrenformer.model import Wrenformer
from aviary.wrenformer.utils import print_walltime

try:
    import wandb
except ImportError:
    pass

__author__ = "Janosh Riebesell"
__date__ = "2022-06-12"

torch.manual_seed(0)  # ensure reproducible results

reg_key, clf_key = "regression", "classification"


@print_walltime(end_desc="run_wrenformer()")
def run_wrenformer(
    run_name: str,
    task_type: TaskType,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    epochs: int,
    timestamp: str = None,
    input_col: str = None,
    id_col: str = "material_id",
    n_attn_layers: int = 4,
    wandb_project: str = None,
    checkpoint: Literal["local", "wandb"] | None = None,
    run_params: dict[str, Any] = None,
    optimizer: str | tuple[str, dict] = "AdamW",
    scheduler: str | tuple[str, dict] = "LambdaLR",
    learning_rate: float = 1e-4,
    batch_size: int = 128,
    warmup_steps: int = 10,
    swa_start: float = 0.7,
    swa_lr: float = None,
    embedding_aggregations: Sequence[str] = ("mean",),
    verbose: bool = False,
) -> tuple[dict[str, float], dict[str, Any], pd.DataFrame]:
    """Run a single matbench task.

    Args:
        run_name (str): Can be any string to describe the Roost/Wren variant being trained.
            Include 'robust' to use a robust loss function and have the model learn to
            predict an aleatoric uncertainty.
        task_type ('regression' | 'classification'): What type of task to train the model for.
        train_df (pd.DataFrame): Dataframe containing the training data.
        test_df (pd.DataFrame): Dataframe containing the test data.
        target_col (str): Name of df column containing the target values.
        input_col (str): Name of df column containing the input values. Defaults to 'wyckoff' if
            'wren' in run_name else 'composition'.
        id_col (str): Name of df column containing material IDs.
        epochs (int): How many epochs to train for. Defaults to 100.
        timestamp (str): Will be included in run_params and used as file name prefix for model
            checkpoints and result files. Defaults to None.
        n_attn_layers (int): Number of transformer encoder layers to use. Defaults to 4.
        wandb_project (str | None): Name of Weights and Biases project where to log this run.
            Defaults to None which means logging is disabled.
        checkpoint (None | 'local' | 'wandb'): Whether to save the model+optimizer+scheduler state
            dicts to disk (local) or upload to WandB. Defaults to None.
            To later copy a wandb checkpoint file to cwd and use it:
            ```py
            run_path = "<user|team>/<project>/<run_id>"  # e.g. aviary/matbench/31qh7b5q
            checkpoint = wandb.restore("checkpoint.pth", run_path)
            torch.load(checkpoint.name)
            ```
        run_params (dict[str, Any]): Additional parameters to merge into the run's dict of
            hyperparams. Will be logged to wandb. Can be anything really. Defaults to {}.
        optimizer (str | tuple[str, dict]): Name of a torch.optim.Optimizer class like 'Adam',
            'AdamW', 'SGD', etc. Can be a string or a string and dict with params to pass to the
            class. Defaults to 'AdamW'.
        scheduler (str | tuple[str, dict]): Name of a torch.optim.lr_scheduler class like
            'LambdaLR', 'StepLR', 'CosineAnnealingLR', etc. Defaults to 'LambdaLR'. Can be a string
            or a string and dict with params to pass to the class. E.g.
            ('CosineAnnealingLR', {'T_max': n_epochs}).
        learning_rate (float): The optimizer's learning rate. Defaults to 1e-4.
        batch_size (int): The mini-batch size during training. Defaults to 128.
        warmup_steps (int): How many warmup steps the scheduler should do. Defaults to 10.
        swa_start (float | None): When to start using stochastic weight averaging during training.
            Should be a float between 0 and 1. 0.7 means start SWA after 70% of epochs. Set to
            None to disable SWA. Defaults to 0.7. Proposed in https://arxiv.org/abs/1803.05407.
        swa_lr (float): Learning rate for SWA scheduler. Defaults to learning_rate.
        embedding_aggregations (list[str]): Aggregations to apply to the learned embedding returned
            by the transformer encoder before passing into the ResidualNetwork. One or more of
            ['mean', 'std', 'sum', 'min', 'max']. Defaults to ['mean'].
        verbose (bool): Whether to print progress and metrics to stdout. Defaults to False.

    Raises:
        ValueError: On unknown dataset_name or invalid checkpoint.

    Returns:
        tuple[dict[str, float], dict[str, Any]]: 1st dict are the model's test set metrics.
            2nd dict are the run's hyperparameters. 3rd is the dataframe with test set predictions.
    """
    if checkpoint not in (None, "local", "wandb"):
        raise ValueError(f"Unknown {checkpoint=}")
    if checkpoint == "wandb" and not wandb_project:
        raise ValueError(f"Cannot save checkpoint to wandb if {wandb_project=}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pytorch running on {device=}")

    if "wren" in run_name.lower():
        input_col = input_col or "wyckoff"
        embedding_type = "wyckoff"
    elif "roost" in run_name.lower():
        input_col = input_col or "composition"
        embedding_type = "composition"
    else:
        raise ValueError(
            f"{run_name = } must contain 'roost' or 'wren' (case insensitive)"
        )

    robust = "robust" in run_name.lower()
    loss_func = (
        (RobustL1Loss if robust else torch.nn.L1Loss())
        if task_type == reg_key
        else (torch.nn.NLLLoss() if robust else torch.nn.CrossEntropyLoss())
    )
    loss_dict = {target_col: (task_type, loss_func)}
    normalizer_dict = {target_col: Normalizer() if task_type == reg_key else None}

    data_loader_kwargs = dict(
        target_col=target_col,
        input_col=input_col,
        id_col=id_col,
        embedding_type=embedding_type,
    )
    train_loader = df_to_in_mem_dataloader(
        train_df, batch_size=batch_size, shuffle=True, **data_loader_kwargs
    )

    test_loader = df_to_in_mem_dataloader(test_df, batch_size=512, **data_loader_kwargs)

    # embedding_len is the length of the embedding vector for a Wyckoff position encoding the
    # element type (usually 200-dim matscholar embeddings) and Wyckoff position (see
    # 'bra-alg-off.json') + 1 for the weight of that Wyckoff position (or element) in the material
    embedding_len = train_loader.tensors[0][0].shape[-1]
    # Roost and Wren embedding size resp.
    assert embedding_len in (200 + 1, 200 + 1 + 444), f"{embedding_len=}"

    model_params = dict(
        # 1 for regression, n_classes for classification
        n_targets=[1 if task_type == reg_key else train_df[target_col].max() + 1],
        n_features=embedding_len,
        task_dict={target_col: task_type},  # e.g. {'exfoliation_en': 'regression'}
        n_attn_layers=n_attn_layers,
        robust=robust,
        embedding_aggregations=embedding_aggregations,
    )
    model = Wrenformer(**model_params)
    model.to(device)
    if isinstance(optimizer, str):
        optimizer_name, optimizer_params = optimizer, None
    elif isinstance(optimizer, (tuple, list)):
        optimizer_name, optimizer_params = optimizer
    else:
        raise ValueError(f"Unknown {optimizer=}")
    optimizer_cls = getattr(torch.optim, optimizer_name)
    optimizer_instance = optimizer_cls(
        params=model.parameters(), lr=learning_rate, **(optimizer_params or {})
    )

    # This lambda goes up linearly until warmup_steps, then follows a power law decay.
    # Acts as a prefactor to the learning rate, i.e. actual_lr = lr_lambda(epoch) *
    # learning_rate.
    if scheduler == "LambdaLR":
        scheduler_name, scheduler_params = "LambdaLR", {
            "lr_lambda": lambda epoch: min(
                (epoch + 1) ** (-0.5), (epoch + 1) * warmup_steps ** (-1.5)
            )
        }
    elif isinstance(scheduler, str):
        scheduler_name, scheduler_params = scheduler, None
    elif isinstance(scheduler, (tuple, list)):
        scheduler_name, scheduler_params = scheduler
    else:
        raise ValueError(f"Unknown {scheduler=}")
    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
    lr_scheduler = scheduler_cls(optimizer_instance, **(scheduler_params or {}))

    if swa_start is not None:
        swa_lr = swa_lr or learning_rate
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer_instance, swa_lr=swa_lr)

    run_params = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "optimizer": {"name": optimizer_name, "params": optimizer_params},
        "lr_scheduler": {"name": scheduler_name, "params": scheduler_params},
        "batch_size": batch_size,
        "n_attn_layers": n_attn_layers,
        "target": target_col,
        "warmup_steps": warmup_steps,
        "robust": robust,
        "embedding_len": embedding_len,
        "losses": loss_dict,
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "trainable_params": model.num_params,
        "task_type": task_type,
        "swa": {
            "start": swa_start,
            "epochs": int(swa_start * epochs),
            "learning_rate": swa_lr,
        }
        if swa_start
        else None,
        "embedding_aggregations": ",".join(embedding_aggregations),
        **(run_params or {}),
    }
    if timestamp:
        run_params["timestamp"] = timestamp
    for x in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        if x in os.environ:
            run_params[x] = os.environ[x]

    if wandb_project:
        if wandb.run is None:
            wandb.login()
        wandb.init(
            project=wandb_project,  # run will be added to this project
            # https://docs.wandb.ai/guides/track/launch#init-start-error
            settings=wandb.Settings(start_method="fork"),
            name=run_name,
            config=run_params,
        )

    for epoch in range(epochs):
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}")

        train_metrics = model.evaluate(
            train_loader,
            loss_dict,
            optimizer_instance,
            normalizer_dict,
            action="train",
            verbose=verbose,
        )

        with torch.no_grad():
            val_metrics = model.evaluate(
                test_loader,
                loss_dict,
                None,
                normalizer_dict,
                action="evaluate",
                verbose=verbose,
            )

        if swa_start and epoch >= int(swa_start * epochs):
            if epoch == int(swa_start * epochs):
                print("Starting stochastic weight averaging...")
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            if scheduler_name == "ReduceLROnPlateau":
                val_metric = val_metrics[target_col][
                    "MAE" if task_type == reg_key else "Accuracy"
                ]
                lr_scheduler.step(val_metric)
            else:
                lr_scheduler.step()

        model.epoch += 1

        if wandb_project:
            wandb.log({"training": train_metrics, "validation": val_metrics})

    # get test set predictions
    if swa_start is not None:
        n_swa_epochs = int((1 - swa_start) * epochs)
        print(
            f"Using SWA model with weights averaged over {n_swa_epochs} epochs ({swa_start = })"
        )

    inference_model = swa_model if swa_start else model
    inference_model.eval()

    with torch.no_grad():
        predictions = torch.cat(
            [inference_model(*inputs)[0] for inputs, *_ in test_loader]
        )

    if robust:
        predictions, aleat_log_std = predictions.chunk(2, dim=1)
        aleat_std = aleat_log_std.exp().cpu().numpy().squeeze()
        test_df["aleat_std"] = aleat_std.tolist()
    if task_type == clf_key:
        predictions = predictions.softmax(dim=1)

    predictions = predictions.cpu().numpy().squeeze()
    targets = test_df[target_col]
    pred_col = f"{target_col}_pred"
    test_df[pred_col] = predictions.tolist()  # requires shuffle=False for test_loader

    test_metrics = get_metrics(targets, predictions, task_type)
    test_metrics["test_size"] = len(test_df)

    # save model checkpoint
    if checkpoint is not None:
        checkpoint_dict = {
            "model_params": model_params,
            "model_state": inference_model.state_dict(),
            "optimizer_state": optimizer_instance.state_dict(),
            "scheduler_state": lr_scheduler.state_dict(),
            "loss_dict": loss_dict,
            "epoch": epochs,
            "metrics": test_metrics,
            "run_name": run_name,
            "normalizer_dict": normalizer_dict,
            "run_params": run_params,
        }
        if checkpoint == "local":
            os.makedirs(f"{ROOT}/models", exist_ok=True)
            checkpoint_path = f"{ROOT}/models/{timestamp}-{run_name}.pth"
            torch.save(checkpoint_dict, checkpoint_path)
        if checkpoint == "wandb":
            assert (
                wandb_project and wandb.run is not None
            ), "can't save model checkpoint to Weights and Biases, wandb.run is None"
            torch.save(checkpoint_dict, f"{wandb.run.dir}/checkpoint.pth")

    # record test set metrics and scatter/ROC plots to wandb
    if wandb_project:
        wandb.run.summary["test"] = test_metrics
        table_cols = [id_col, target_col, pred_col]
        if robust:
            table_cols.append("aleat_std")
        table = wandb.Table(dataframe=test_df[table_cols])
        wandb.log({"test_set_predictions": table})
        if task_type == reg_key:
            from sklearn.metrics import r2_score

            MAE = np.abs(targets - predictions).mean()
            R2 = r2_score(targets, predictions)
            title = f"{run_name}\n{MAE=:.2f}\n{R2=:.2f}"
            scatter_plot = wandb.plot.scatter(table, target_col, pred_col, title=title)
            wandb.log({"true_pred_scatter": scatter_plot})
        elif task_type == clf_key:
            from sklearn.metrics import accuracy_score, roc_auc_score

            ROCAUC = roc_auc_score(targets, predictions[:, 1])
            accuracy = accuracy_score(targets, predictions.argmax(axis=1))
            title = f"{run_name}\n{accuracy=:.2f}\n{ROCAUC=:.2f}"
            roc_curve = wandb.plot.roc_curve(targets, predictions)
            wandb.log({"roc_curve": roc_curve})

        wandb.finish()

    return test_metrics, run_params, test_df
