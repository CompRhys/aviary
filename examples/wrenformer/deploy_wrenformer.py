from __future__ import annotations

import os
from typing import Any, Literal

import pandas as pd
import torch
import wandb
from tqdm import tqdm

from aviary import ROOT
from aviary.core import BaseModelClass
from aviary.utils import get_metrics
from aviary.wrenformer.data import df_to_in_mem_dataloader
from aviary.wrenformer.model import Wrenformer

__author__ = "Janosh Riebesell"
__date__ = "2022-08-25"


def make_ensemble_predictions(
    checkpoint_paths: list[str],
    df: pd.DataFrame,
    target_col: str = None,
    input_col: str = "wyckoff",
    model_class: type[BaseModelClass] = Wrenformer,
    device: str = None,
    print_metrics: bool = True,
    warn_target_mismatch: bool = False,
    embedding_type: Literal["wyckoff", "composition"] = "wyckoff",
    batch_size: int = 512,
    pbar: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Make predictions using an ensemble of Wrenformer models.

    Args:
        checkpoint_paths (list[str]): File paths to model checkpoints created with torch.save().
        df (pd.DataFrame): Dataframe to make predictions on. Will be returned with additional
            columns holding model predictions (and uncertainties for robust models) for each
            model checkpoint.
        target_col (str): Column holding target values. Defaults to None. If None, will not print
            performance metrics.
        input_col (str, optional): Column holding input values. Defaults to 'wyckoff'.
        device (str, optional): torch.device. Defaults to "cuda" if torch.cuda.is_available()
            else "cpu".
        print_metrics (bool, optional): Whether to print performance metrics. Defaults to True
            if target_col is not None.
        warn_target_mismatch (bool, optional): Whether to warn if target_col != target_name from
            model checkpoint. Defaults to False.
        embedding_type ('wyckoff' | 'composition', optional): Type of embedding to use depending on
            using Wren-/Roostformer ensemble. Defaults to "wyckoff".
        batch_size (int, optional): Batch size for data loader. Defaults to 512. Can be large to
            speedup inference.

    Returns:
        pd.DataFrame: Input dataframe with added columns for model and ensemble predictions. If
            target_col is not None, returns a 2nd dataframe containing model and ensemble metrics.
    """
    # TODO: Add support for predicting all tasks a multi-task models was trained on. Currently only
    # handles single targets. Low priority as multi-tasking is rarely used.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pytorch running on {device=}")

    data_loader = df_to_in_mem_dataloader(
        df=df,
        target_col=target_col,
        input_col=input_col,
        batch_size=batch_size,
        embedding_type=embedding_type,
    )

    # tqdm(disable=None) means suppress output in non-tty (e.g. CI/log files) but keep in
    # terminal (i.e. tty mode) https://git.io/JnBOi
    for idx, checkpoint_path in tqdm(
        enumerate(tqdm(checkpoint_paths), 1), disable=None if pbar else True
    ):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_params = checkpoint["model_params"]
        target_name, task_type = list(model_params["task_dict"].items())[0]
        assert task_type in ("regression", "classification"), f"invalid {task_type = }"
        if warn_target_mismatch and target_name != target_col:
            print(
                f"Warning: {target_col = } does not match {target_name = } in checkpoint. "
                "If this is not by accident, disable this warning by passing warn_target=False."
            )
        model = model_class(**model_params)
        model.to(device)

        model.load_state_dict(checkpoint["model_state"])

        with torch.no_grad():
            predictions = torch.cat([model(*inputs)[0] for inputs, *_ in data_loader])

        if model.robust:
            predictions, aleat_log_std = predictions.chunk(2, dim=1)
            aleat_std = aleat_log_std.exp().cpu().numpy().squeeze()
            df[f"aleatoric_std_{idx}"] = aleat_std.tolist()

        predictions = predictions.cpu().numpy().squeeze()
        pred_col = f"{target_col}_pred_{idx}" if target_col else f"pred_{idx}"
        df[pred_col] = predictions.tolist()

    df_preds = df.filter(regex=r"_pred_\d")
    df[f"{target_col}_pred_ens"] = ensemble_preds = df_preds.mean(axis=1)
    df[f"{target_col}_epistemic_std_ens"] = epistemic_std = df_preds.std(axis=1)

    if df.columns.str.startswith("aleatoric_std_").sum() > 0:
        aleatoric_std = df.filter(regex=r"aleatoric_std_\d").mean(axis=1)
        df[f"{target_col}_aleatoric_std_ens"] = aleatoric_std
        df[f"{target_col}_total_std_ens"] = (
            epistemic_std**2 + aleatoric_std**2
        ) ** 0.5

    if target_col and print_metrics:
        targets = df[target_col]
        all_model_metrics = pd.DataFrame(
            [
                get_metrics(targets, df_preds[pred_col], task_type)
                for pred_col in df_preds
            ],
            index=df_preds.columns,
        )

        print("\nSingle model performance:")
        print(all_model_metrics.describe().round(4).loc[["mean", "std"]])

        ensemble_metrics = get_metrics(targets, ensemble_preds, task_type)

        print("\nEnsemble performance:")
        for key, val in ensemble_metrics.items():
            print(f"{key:<8} {val:.3}")
        return df, all_model_metrics

    return df


def deploy_wandb_checkpoints(
    runs: list[wandb.sdk.wandb_run.Run],
    df: pd.DataFrame,
    input_col: str,
    target_col: str,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Function that downloads and caches checkpoints for an ensemble of Wrenformer models, then
    makes predictions on some dataset, prints ensemble metrics and stores predictions to CSV.

    Args:
        runs (list[wandb.sdk.wandb_run.Run]): List of WandB runs to download model checkpoints from
            which are then loaded into memory to generate predictions for the input_col in df.
        df (pd.DataFrame): Test dataset on which to make predictions.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Original input dataframe with added columns for model
            predictions and uncertainties. The 2nd dataframe holds ensemble performance metrics
            like mean and standard deviation of MAE/RMSE.
    """
    print(
        f"Loading checkpoints for the following {len(runs)} run ID(s):\n"
        f"{', '.join(run.id for run in runs)}\n"
    )

    checkpoint_paths: list[str] = []
    for run in runs:
        run_path = "/".join(run.path)
        checkpoint_dir = f"{ROOT}/.wandb_checkpoints/{run_path}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = f"{checkpoint_dir}/checkpoint.pth"
        checkpoint_paths.append(checkpoint_path)

        # download checkpoint from wandb if not already present
        if os.path.isfile(checkpoint_path):
            continue
        wandb.restore("checkpoint.pth", root=checkpoint_dir, run_path=run_path)

    df, ensemble_metrics = make_ensemble_predictions(
        checkpoint_paths, df=df, input_col=input_col, target_col=target_col, **kwargs
    )

    return df, ensemble_metrics
