from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import torch
import wandb.apis.public
from torch.utils.data import DataLoader
from tqdm import tqdm

from aviary.core import BaseModelClass
from aviary.data import InMemoryDataLoader
from aviary.utils import get_metrics, print_walltime

__author__ = "Janosh Riebesell"
__date__ = "2022-08-25"


def make_ensemble_predictions(
    checkpoint_paths: list[str],
    data_loader: DataLoader | InMemoryDataLoader,
    model_cls: type[BaseModelClass],
    df: pd.DataFrame,
    target_col: str = None,
    device: str = None,
    print_metrics: bool = True,
    warn_target_mismatch: bool = False,
    pbar: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Make predictions using an ensemble of models.

    Args:
        checkpoint_paths (list[str]): File paths to model checkpoints created with torch.save().
        data_loader (DataLoader | InMemoryDataLoader): Data loader to use for predictions.
        model_cls (type[BaseModelClass]): Model class to use for predictions.
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
        pbar (bool, optional): Whether to show progress bar running over checkpoints.
            Defaults to True.

    Returns:
        pd.DataFrame: Input dataframe with added columns for model and ensemble predictions. If
            target_col is not None, returns a 2nd dataframe containing model and ensemble metrics.
    """
    # TODO: Add support for predicting all tasks a multi-task models was trained on. Currently only
    # handles single targets. Low priority as multi-tasking is rarely used.
    if not checkpoint_paths:
        raise ValueError(f"{checkpoint_paths=} must not be empty")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # tqdm(disable=None) means suppress output in CI/log files but keep in terminal
    # (i.e. tty mode) https://git.io/JnBOi
    print(f"Pytorch running on {device=}")
    for idx, checkpoint_path in tqdm(
        enumerate(tqdm(checkpoint_paths), 1), disable=None if pbar else True
    ):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {checkpoint_path=}") from exc

        model_params = checkpoint.get("model_params")
        if model_params is None:
            raise ValueError(f"model_params not found in {checkpoint_path=}")

        target_name, task_type = list(model_params["task_dict"].items())[0]
        assert task_type in ("regression", "classification"), f"invalid {task_type = }"
        if warn_target_mismatch and target_name != target_col:
            print(
                f"Warning: {target_col = } does not match {target_name = } in checkpoint. If this "
                "is not by accident, disable this warning by passing warn_target_mismatch=False."
            )
        model = model_cls(**model_params)
        model.to(device)

        model.load_state_dict(checkpoint["model_state"])

        with torch.no_grad():
            preds = np.concatenate(
                [model(*inputs)[0].cpu().numpy() for inputs, *_ in data_loader]
            ).squeeze()

        if model.robust:
            preds, aleat_log_std = preds.T
            df[f"aleatoric_std_{idx}"] = aleatoric_std = np.exp(aleat_log_std)

        pred_col = f"{target_col}_pred_{idx}" if target_col else f"pred_{idx}"
        df[pred_col] = preds

    df_preds = df.filter(regex=r"_pred_\d")
    df[f"{target_col}_pred_ens"] = ensemble_preds = df_preds.mean(axis=1)
    df[f"{target_col}_epistemic_std_ens"] = epistemic_std = df_preds.std(axis=1)

    if df.columns.str.startswith("aleatoric_std_").any():
        aleatoric_std = df.filter(regex=r"aleatoric_std_\d").mean(axis=1)
        df[f"{target_col}_aleatoric_std_ens"] = aleatoric_std
        df[f"{target_col}_total_std_ens"] = (
            epistemic_std**2 + aleatoric_std**2
        ) ** 0.5

    if target_col:
        targets = df[target_col]
        all_model_metrics = [
            get_metrics(targets, df_preds[col], task_type) for col in df_preds
        ]
        df_metrics = pd.DataFrame(all_model_metrics, index=list(df_preds))

        if print_metrics:
            print("\nSingle model performance:")
            print(df_metrics.describe().round(4).loc[["mean", "std"]])

            ensemble_metrics = get_metrics(targets, ensemble_preds, task_type)

            print("\nEnsemble performance:")
            for key, val in ensemble_metrics.items():
                print(f"{key:<8} {val:.3}")
        return df, df_metrics

    return df


@print_walltime(end_desc="predict_from_wandb_checkpoints")
def predict_from_wandb_checkpoints(
    runs: list[wandb.apis.public.Run], cache_dir: str, **kwargs: Any
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Function that downloads and caches checkpoints for an ensemble of models, then
    makes predictions on some dataset, prints ensemble metrics and stores predictions to CSV.

    Args:
        runs (list[wandb.apis.public.Run]): List of WandB runs to download model checkpoints from
            which are then loaded into memory to generate predictions for the input_col in df.
        cache_dir (str): Directory to cache downloaded checkpoints in.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Original input dataframe with added columns for model
            predictions and uncertainties. The 2nd dataframe holds ensemble performance metrics
            like mean and standard deviation of MAE/RMSE.
    """
    print(f"Using checkpoints from {len(runs)} run(s):")

    run_target = runs[0].config["target"]
    assert all(
        run_target == run.config["target"] for run in runs
    ), f"Runs have differing targets, first {run_target=}"

    target_col = kwargs.get("target_col")
    if target_col and target_col != run_target:
        print(f"\nWarning: {target_col=} does not match {run_target=}")

    checkpoint_paths: list[str] = []

    for idx, run in enumerate(runs, 1):
        run_path = "/".join(run.path)
        out_dir = f"{cache_dir}/{run_path}"
        os.makedirs(out_dir, exist_ok=True)

        checkpoint_path = f"{out_dir}/checkpoint.pth"
        checkpoint_paths.append(checkpoint_path)
        print(f"{idx:>3}/{len(runs)}: {run.url}\n\t{checkpoint_path}")

        with open(f"{out_dir}/run.md", "w") as md_file:
            md_file.write(f"[{run.name}]({run.url})\n")

        if not os.path.isfile(checkpoint_path):
            run.file("checkpoint.pth").download(root=out_dir)

    df, ensemble_metrics = make_ensemble_predictions(checkpoint_paths, **kwargs)

    # round to save disk space and speed up cloud storage uploads
    return df.round(6), ensemble_metrics
