import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from aviary.core import BaseModelClass, Normalizer, sampled_softmax
from aviary.data import InMemoryDataLoader
from aviary.utils import get_metrics


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
        checkpoint_paths (list[str]): File paths to model checkpoints created with
            torch.save().
        data_loader (DataLoader | InMemoryDataLoader): Data loader to use for predictions.
        model_cls (type[BaseModelClass]): Model class to use for predictions.
        df (pd.DataFrame): Dataframe to make predictions on. Will be returned with
            additional columns holding model predictions (and uncertainties for robust
            columns holding model predictions (and uncertainties for robust models)
            for each model checkpoint.
        target_col (str): Column holding target values. Defaults to None. If None, will
            not print performance metrics.
        device (str, optional): torch.device. Defaults to "cuda" if
            torch.cuda.is_available() else "cpu".
        print_metrics (bool, optional): Whether to print performance metrics. Defaults
            to True if target_col is not None.
        warn_target_mismatch (bool, optional): Whether to warn if target_col !=
            target_name from model checkpoint. Defaults to False.
        pbar (bool, optional): Whether to show progress bar running over checkpoints.
            Defaults to True.

    Returns:
        pd.DataFrame: Input dataframe with added columns for model and ensemble
            predictions. If target_col is not None, returns a 2nd dataframe
            containing model and ensemble metrics.
    """
    # TODO: Add support for predicting all tasks a multi-task models was trained on.
    # Currently only handles single targets. Low priority as multi-tasking is rarely used.
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
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load {checkpoint_path=}") from exc

        model_params = checkpoint.get("model_params")
        if model_params is None:
            raise ValueError(f"model_params not found in {checkpoint_path=}")

        target_name, task_type = next(iter(model_params["task_dict"].items()))
        assert task_type in ("regression", "classification"), f"invalid {task_type = }"
        if warn_target_mismatch and target_name != target_col:
            print(
                f"Warning: {target_col = } does not match {target_name = } in "
                f"{checkpoint_path=}. If this is not by accident, disable this warning "
                "by passing warn_target_mismatch=False."
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

        pred_col = f"{target_col}_pred_{idx}" if target_col else f"pred_{idx}"

        if model.robust:
            if task_type == "regression":
                preds, aleat_log_std = preds.T
                ale_col = (
                    f"{target_col}_aleatoric_std_{idx}"
                    if target_col
                    else f"aleatoric_std_{idx}"
                )
                df[pred_col] = preds
                df[ale_col] = aleatoric_std = np.exp(aleat_log_std)
            elif task_type == "classification":
                # need to convert to tensor to use `sampled_softmax`
                preds = torch.from_numpy(preds).to(device)
                pre_logits, log_std = preds.chunk(2, dim=1)
                logits = sampled_softmax(pre_logits, log_std)
                df[pred_col] = logits.argmax(dim=1).cpu().numpy()
        else:
            if task_type == "regression":
                df[pred_col] = preds
            else:
                logits = softmax(preds, dim=1)
                df[pred_col] = logits.argmax(dim=1).cpu().numpy()

    # denormalize predictions if a normalizer was used during training
    if checkpoint["normalizer_dict"][target_name] is not None:
        assert task_type == "regression", "Normalization only takes place for regression."
        normalizer = Normalizer.from_state_dict(
            checkpoint["normalizer_dict"][target_name]
        )
        mean = normalizer.mean.cpu().numpy()
        std = normalizer.std.cpu().numpy()
        # denorm the mean and aleatoric uncertainties separately
        df[pred_col] = df[pred_col] * std + mean
        if model.robust:
            df[ale_col] = df[ale_col] * std

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
