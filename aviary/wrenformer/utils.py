from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Generator, Literal

import pandas as pd
import torch
from tqdm import tqdm

from aviary.core import BaseModelClass
from aviary.utils import get_metrics
from aviary.wrenformer.data import df_to_in_mem_dataloader
from aviary.wrenformer.model import Wrenformer

__author__ = "Janosh Riebesell"
__date__ = "2022-05-10"


def _int_keys(dct: dict) -> dict:
    # JSON stringifies all dict keys during serialization and does not revert
    # back to floats and ints during parsing. This json.load() hook converts keys
    # containing only digits to ints.
    return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in dct.items()}


def recursive_dict_merge(d1: dict, d2: dict) -> dict:
    """Merge two dicts recursively."""
    for key in d2:
        if key in d1 and isinstance(d1[key], dict) and isinstance(d2[key], dict):
            recursive_dict_merge(d1[key], d2[key])
        else:
            d1[key] = d2[key]
    return d1


def merge_json_on_disk(
    dct: dict,
    file_path: str,
    on_non_serializable: Literal["annotate", "error"] = "annotate",
) -> None:
    """Merge a dict into a (possibly) existing JSON file.

    Args:
        file_path (str): Path to JSON file. File will be created if not exist.
        dct (dict): Dictionary to merge into JSON file.
        on_non_serializable ('annotate' | 'error'): What to do with non-serializable values
            encountered in dct. 'annotate' will replace the offending object with a string
            indicating the type, e.g. '<not serializable: function>'. 'error' will raise
            'TypeError: Object of type function is not JSON serializable'. Defaults to 'annotate'.
    """
    try:
        with open(file_path) as json_file:
            data = json.load(json_file, object_hook=_int_keys)

        dct = recursive_dict_merge(data, dct)
    except (FileNotFoundError, json.decoder.JSONDecodeError):  # file missing or empty
        pass

    def non_serializable_handler(obj: object) -> str:
        # replace functions and classes in dct with string indicating it's a non-serializable type
        return f"<not serializable: {type(obj).__qualname__}>"

    with open(file_path, "w") as file:
        default = (
            non_serializable_handler if on_non_serializable == "annotate" else None
        )
        json.dump(dct, file, default=default, indent=2)


@contextmanager
def print_walltime(
    start_desc: str = "",
    end_desc: str = "",
    newline: bool = True,
) -> Generator[None, None, None]:
    """Context manager and decorator that prints the wall time of its lifetime.

    Args:
        start_desc (str): Text to print when entering context. Defaults to ''.
        end_desc (str): Text to print when exiting context. Will be followed by 'took
            {duration} sec'. i.e. f"{end_desc} took 1.23 sec". Defaults to ''.
        newline (bool): Whether to print a newline after start_desc. Defaults to True.
    """
    start_time = time.perf_counter()
    if start_desc:
        print(start_desc, end="\n" if newline else "")

    try:
        yield
    finally:
        run_time = time.perf_counter() - start_time
        print(f"{end_desc} took {run_time:.2f} sec")


def make_ensemble_predictions(
    checkpoint_paths: list[str],
    df: pd.DataFrame,
    target_col: str = None,
    input_col: str = "wyckoff",
    model_class: type[BaseModelClass] = Wrenformer,
    device: str = None,
    print_metrics: bool = True,
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

    Returns:
        pd.DataFrame: Input dataframe with added columns for model and ensemble predictions. If
            target_col is not None, returns a 2nd dataframe containing model and ensemble metrics.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = df_to_in_mem_dataloader(
        df=df,
        target_col=target_col,
        input_col=input_col,
        batch_size=512,
        embedding_type="wyckoff",
    )

    print(f"Predicting with {len(checkpoint_paths):,} model checkpoints(s)")

    for idx, checkpoint_path in enumerate(tqdm(checkpoint_paths), 1):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model_params = checkpoint["model_params"]
        target_name, task_type = next(model_params["task_dict"].items())
        assert task_type in ("regression", "classification"), f"invalid {task_type = }"
        if target_name != target_col:
            print(
                f"Warning: {target_col = } does not match {target_name = } in checkpoint."
            )
        model = model_class(**model_params)
        model.to(device)

        model.load_state_dict(checkpoint["model_state"])

        with torch.no_grad():
            predictions = torch.cat([model(*inputs)[0] for inputs, *_ in data_loader])

        if model.robust:
            predictions, aleat_log_std = predictions.chunk(2, dim=1)
            aleat_std = aleat_log_std.exp().cpu().numpy().squeeze()
            df[f"aleat_std_{idx}"] = aleat_std.tolist()

        predictions = predictions.cpu().numpy().squeeze()
        pred_col = f"{target_col}_pred_{idx}" if target_col else f"pred_{idx}"
        df[pred_col] = predictions.tolist()

    df_preds = df.filter(regex=r"_pred_\d")
    df[f"{target_col}_pred_ens"] = ensemble_preds = df_preds.mean(axis=1)
    df[f"{target_col}_ens_epistemic_std"] = df_preds.std(axis=1)

    if target_col and print_metrics:
        targets = df[target_col]
        all_model_metrics = pd.DataFrame(
            [
                get_metrics(targets, df_preds[pred_col], task_type)
                for pred_col in df_preds
            ],
            index=df_preds.columns,
        )

        print("Single model performance:")
        print(all_model_metrics.describe().loc[["mean", "std"]])

        ensemble_metrics = get_metrics(targets, ensemble_preds, task_type)

        print("Ensemble performance:")
        for key, val in ensemble_metrics.items():
            print(f"{key:<8} {val:.3}")
        return df, all_model_metrics

    return df
