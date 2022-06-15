from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from torch.optim.swa_utils import SWALR, AveragedModel
from tqdm import tqdm

from aviary import ROOT
from aviary.core import Normalizer, TaskType
from aviary.data import InMemoryDataLoader
from aviary.losses import RobustL1Loss
from aviary.utils import get_metrics
from aviary.wrenformer.data import (
    collate_batch,
    get_composition_embedding,
    wyckoff_embedding_from_aflow_str,
)
from aviary.wrenformer.model import Wrenformer
from aviary.wrenformer.utils import print_walltime

__author__ = "Janosh Riebesell"
__date__ = "2022-06-12"

torch.manual_seed(0)  # ensure reproducible results

reg_key, clf_key = "regression", "classification"
learning_rate = 1e-4
warmup_steps = 10


def lr_lambda(epoch: int) -> float:
    """Learning rate schedule. Goes up linearly until warmup_steps, then follows a
    power law decay.
    """
    return min((epoch + 1) ** (-0.5), (epoch + 1) * warmup_steps ** (-1.5))


@print_walltime(end_desc="run_wrenformer()")
def run_wrenformer(
    run_name: str,
    task_type: TaskType,
    timestamp: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    id_col: str = "material_id",
    epochs: int = 100,
    n_attn_layers: int = 4,
    wandb_project: str = None,
    checkpoint: Literal["local", "wandb"] | None = None,
    swa: bool = True,
    run_params: dict[str, Any] = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Run a single matbench task.

    Args:
        run_name (str): Can be any string to describe the Roost/Wren variant being trained.
            Include 'robust' to use a robust loss function and have the model learn to
            predict an aleatoric uncertainty.
        task_type ('regression' | 'classification'): What type of task to train the model for.
        timestamp (str): Will be used as prefix for model checkpoints and result files.
        train_df (pd.DataFrame): Dataframe containing the training data.
        test_df (pd.DataFrame): Dataframe containing the test data.
        target_col (str): Name of df column containing the target values.
        id_col (str): Name of df column containing material IDs.
        epochs (int): How many epochs to train for. Defaults to 100.
        n_attn_layers (int): Number of transformer encoder layers to use. Defaults to 4.
        wandb_project (str | None): Name of Weights and Biases project where to log this run.
            Defaults to None which means logging is disabled.
        checkpoint (None | 'local' | 'wandb'): Whether to save the model+optimizer+scheduler state
            dicts to disk (local) or upload to wandb. Defaults to None.
            To later copy a wandb checkpoint file to cwd and use it:
            ```py
            run_path="<user|team>/<project>/<run_id>"  # e.g. aviary/matbench/31qh7b5q
            checkpoint = wandb.restore("checkpoint.pth", run_path)
            torch.load(checkpoint.name)
            ```
        swa (bool): Whether to use stochastic weight averaging for training and inference.
            Defaults to True.
        run_params (dict[str, Any]): Additional parameters to merge into the run's dict of
            hyperparams. Will be logged to wandb. Can be anything really. Defaults to {}.

    Raises:
        ValueError: On unknown dataset_name or invalid checkpoint.

    Returns:
        tuple[dict[str, float], dict[str, Any]]: 1st dict are the model's test set metrics.
            2nd dict are the run's hyperparameters.
    """
    if checkpoint not in (None, "local", "wandb"):
        raise ValueError(f"Unknown {checkpoint=}")
    if checkpoint == "wandb" and not wandb_project:
        raise ValueError(f"Cannot save checkpoint to wandb if {wandb_project=}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pytorch running on {device=}")

    for label, df in [("training set", train_df), ("test set", test_df)]:
        if "wren" in run_name.lower():
            err_msg = "Missing 'wyckoff' column in dataframe. "
            err_msg += (
                "Please generate Aflow Wyckoff labels ahead of time."
                if "structure" in df
                else "Trying to deploy Wrenformer on composition-only task?"
            )
            assert "wyckoff" in df, err_msg
            with print_walltime(
                start_desc=f"{label} Generating Wyckoff embeddings", newline=False
            ):
                df["features"] = df.wyckoff.map(wyckoff_embedding_from_aflow_str)
        elif "roost" in run_name.lower():
            df["features"] = df.composition.map(get_composition_embedding)
        else:
            raise ValueError(f"{run_name = } must contain 'roost' or 'wren'")

    robust = "robust" in run_name.lower()
    loss_func = (
        (RobustL1Loss if robust else nn.L1Loss())
        if task_type == reg_key
        else (nn.NLLLoss() if robust else nn.CrossEntropyLoss())
    )
    loss_dict = {target_col: (task_type, loss_func)}
    normalizer_dict = {target_col: Normalizer() if task_type == reg_key else None}

    features, targets, ids = (train_df[x] for x in ["features", target_col, id_col])
    targets = torch.tensor(targets, device=device)
    if targets.dtype == torch.bool:
        targets = targets.long()
    inputs = np.empty(len(features), dtype=object)
    for idx, tensor in enumerate(features):
        inputs[idx] = tensor.to(device)

    train_loader = InMemoryDataLoader(
        [inputs, targets, ids], batch_size=128, shuffle=True, collate_fn=collate_batch
    )

    features, targets, ids = (test_df[x] for x in ["features", target_col, id_col])
    targets = torch.tensor(targets, device=device)
    if targets.dtype == torch.bool:
        targets = targets.long()
    inputs = np.empty(len(features), dtype=object)
    for idx, tensor in enumerate(features):
        inputs[idx] = tensor.to(device)

    test_loader = InMemoryDataLoader(
        [inputs, targets, ids], batch_size=512, collate_fn=collate_batch
    )

    # n_features is the length of the embedding vector for a Wyckoff position encoding
    # the element type (usually 200-dim matscholar embeddings) and Wyckoff position (see
    # 'bra-alg-off.json') + 1 for the weight of that element/Wyckoff position in the
    # material's composition
    n_features = features[0].shape[-1]
    assert n_features in (200 + 1, 200 + 1 + 444)  # Roost and Wren embedding size resp.

    model = Wrenformer(
        n_targets=[1 if task_type == reg_key else 2],
        n_features=n_features,
        task_dict={target_col: task_type},  # e.g. {'exfoliation_en': 'regression'}
        n_attn_layers=n_attn_layers,
        robust=robust,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    swa_model = AveragedModel(model)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    # epoch to start using the SWA model
    swa_start = epochs // 2  # start at 50% of epochs
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    run_params = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "n_attn_layers": n_attn_layers,
        "target": target_col,
        "warmup_steps": warmup_steps,
        "robust": robust,
        "n_features": n_features,  # embedding size
        "losses": str(loss_dict),
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        **(run_params or {}),
    }

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

    for epoch in tqdm(range(epochs)):
        train_metrics = model.evaluate(
            train_loader, loss_dict, optimizer, normalizer_dict, action="train"
        )

        with torch.no_grad():
            val_metrics = model.evaluate(
                test_loader, loss_dict, None, normalizer_dict, action="evaluate"
            )

        if swa and epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        scheduler.step()
        model.epoch += 1

        if wandb_project:
            wandb.log({"training": train_metrics, "validation": val_metrics})

    # get test set predictions
    swa_model.eval()
    model.eval()
    inference_model = swa_model if swa else model

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
    targets = targets.cpu().numpy()
    test_df[(pred_col := "predictions")] = predictions.tolist()

    test_metrics = get_metrics(targets, predictions, task_type)
    test_metrics["size"] = len(test_df)

    # save model checkpoint
    if checkpoint is not None:
        state_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss_dict": loss_dict,
            "epoch": epochs,
            "metrics": test_metrics,
        }
        if checkpoint == "local":
            os.makedirs(f"{ROOT}/models", exist_ok=True)
            checkpoint_path = f"{ROOT}/models/{timestamp}-{run_name}.pth"
            torch.save(state_dict, checkpoint_path)
        if checkpoint == "wandb":
            assert wandb_project and wandb.run is not None, "wandb.run is None"
            torch.save(state_dict, f"{wandb.run.dir}/checkpoint.pth")

    # record test set metrics and scatter/ROC plots to wandb
    if wandb_project:
        wandb.run.summary["test"] = test_metrics
        table_cols = ["mbid", target_col, pred_col]
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

    return test_metrics, run_params
