# ruff: noqa: E501
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aviary.core import Normalizer
from aviary.utils import get_metrics, print_walltime

if TYPE_CHECKING:
    import wandb.apis.public
    from torch.utils.data import DataLoader

    from aviary.core import BaseModelClass
    from aviary.data import InMemoryDataLoader

__author__ = "Janosh Riebesell"
__date__ = "2022-08-25"


def make_ensemble_predictions(
    checkpoint_paths: list[str],
    data_loader: DataLoader | InMemoryDataLoader,
    model_cls: type[BaseModelClass],
    df: pd.DataFrame,
    target_col: str | None = None,
    device: str | None = None,
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
        enumerate(tqdm(checkpoint_paths), start=1), disable=None if pbar else True
    ):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except Exception as exc:
            raise RuntimeError(f"Failed to load {checkpoint_path=}") from exc

        model_params = checkpoint.get("model_params")
        if model_params is None:
            raise ValueError(f"model_params not found in {checkpoint_path=}")

        target_name, task_type = next(iter(model_params["task_dict"].items()))
        assert task_type in ("regression", "classification"), f"invalid {task_type = }"
        if warn_target_mismatch and target_name != target_col:
            print(
                f"Warning: {target_col = } does not match {target_name = } in checkpoint. If this "
                "is not by accident, disable this warning by passing warn_target_mismatch=False."
            )
        model = model_cls(**model_params)
        model.to(device)

        # some models save the state dict under a different key
        state_dict_field = "model_state" if "model_state" in checkpoint else "state_dict"
        model.load_state_dict(checkpoint[state_dict_field])

        with torch.no_grad():
            preds = np.concatenate(
                [model(*inputs)[0].cpu().numpy() for inputs, *_ in data_loader]
            ).squeeze()

        # denormalize predictions if a normalizer was used during training
        if "normalizer_dict" in checkpoint:
            assert task_type == "regression", "Normalization only takes place for regression."
            normalizer = Normalizer.from_state_dict(
                checkpoint["normalizer_dict"][target_name]
            )
            if model.robust:
                # denorm the mean and aleatoroc uncertainties separately
                mean, log_std = np.split(preds, 2, axis=1)
                preds = normalizer.denorm(mean)
                ale_std = np.exp(log_std) * normalizer.std
                preds = np.column_stack([preds, ale_std])
            else:
                preds = normalizer.denorm(preds)

        pred_col = f"{target_col}_pred_{idx}" if target_col else f"pred_{idx}"

        if model.robust:
            preds, aleat_log_std = preds.T
            ale_col = (
                f"{target_col}_aleatoric_std_{idx}"
                if target_col
                else f"aleatoric_std_{idx}"
            )
            df[pred_col] = preds
            df[ale_col] = aleatoric_std = np.exp(aleat_log_std)
        else:
            df[pred_col] = preds

    df_preds = df.filter(regex=r"_pred_\d")

    if len(checkpoint_paths) > 1:
        pred_ens_col = f"{target_col}_pred_ens" if target_col else "pred_ens"
        df[pred_ens_col] = ensemble_preds = df_preds.mean(axis=1)

        pred_epi_std_ens = (
            f"{target_col}_epistemic_std_ens" if target_col else "epistemic_std_ens"
        )
        df[pred_epi_std_ens] = epistemic_std = df_preds.std(axis=1)

        if df.columns.str.startswith("aleatoric_std_").any():
            pred_ale_std_ens = (
                f"{target_col}_aleatoric_std_ens" if target_col else "aleatoric_std_ens"
            )
            pred_tot_std_ens = (
                f"{target_col}_total_std_ens" if target_col else "total_std_ens"
            )
            df[pred_ale_std_ens] = aleatoric_std = df.filter(
                regex=r"aleatoric_std_\d"
            ).mean(axis=1)
            df[pred_tot_std_ens] = (epistemic_std**2 + aleatoric_std**2) ** 0.5

    if target_col is not None:
        targets = df[target_col]
        all_model_metrics = [
            get_metrics(targets, df_preds[col], task_type) for col in df_preds
        ]
        df_metrics = pd.DataFrame(all_model_metrics, index=list(df_preds))

        if print_metrics:
            print("\nSingle model performance:")
            print(df_metrics.describe().round(4).loc[["mean", "std"]])

            if len(checkpoint_paths) > 1:
                ensemble_metrics = get_metrics(targets, ensemble_preds, task_type)

                print("\nEnsemble performance:")
                for key, val in ensemble_metrics.items():
                    print(f"{key:<8} {val:.3}")
        return df, df_metrics

    return df


@print_walltime(end_desc="predict_from_wandb_checkpoints")
def predict_from_wandb_checkpoints(
    runs: list[wandb.apis.public.Run],
    checkpoint_filename: str = "checkpoint.pth",
    cache_dir: str = "./checkpoint_cache",
    **kwargs: Any,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Download and cache checkpoints for an ensemble of models, then make
    predictions on some dataset. Finally print ensemble metrics and store
    predictions to CSV.

    Args:
        runs (list[wandb.apis.public.Run]): List of WandB runs to download model
            checkpoints from which are then loaded into memory to generate
            predictions for the input_col in df.
        checkpoint_filename (str): Name of the checkpoint file to download.
        cache_dir (str): Directory to cache downloaded checkpoints in.
        **kwargs: Additional keyword arguments to pass to make_ensemble_predictions().

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]: Original input dataframe
            with added columns for model predictions and uncertainties. The optional
            2nd dataframe holds ensemble performance metrics like mean and standard
            deviation of MAE/RMSE.
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

    for idx, run in enumerate(runs, start=1):
        run_path = "/".join(run.path)
        out_dir = f"{cache_dir}/{run_path}"
        os.makedirs(out_dir, exist_ok=True)

        checkpoint_path = f"{out_dir}/{checkpoint_filename}"
        checkpoint_paths.append(checkpoint_path)
        print(f"{idx:>3}/{len(runs)}: {run.url}\n\t{checkpoint_path}\n")

        with open(f"{out_dir}/run.md", "w") as md_file:
            md_file.write(f"[{run.name}]({run.url})\n")

        if not os.path.isfile(checkpoint_path):
            run.file(f"{checkpoint_filename}").download(root=out_dir)

    if target_col is not None:
        df, ensemble_metrics = make_ensemble_predictions(checkpoint_paths, **kwargs)
        # round to save disk space and speed up cloud storage uploads
        return df.round(6), ensemble_metrics

    df = make_ensemble_predictions(checkpoint_paths, **kwargs)
    return df.round(6)
