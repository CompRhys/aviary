from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from aviary import ROOT
from examples.wrenformer import train_wrenformer

__author__ = "Janosh Riebesell"
__date__ = "2022-06-13"


def train_wrenformer_on_mp(
    model_name: str,
    df_or_path: str | pd.DataFrame,
    target_col: str,
    id_col: str = "material_id",
    folds: tuple[int, int] | None = None,
    test_size: float | None = None,
    **kwargs,
) -> dict[str, float]:
    """Run a single matbench task.

    Args:
        model_name (str): Can be any string to describe particular Roost/Wren variants. Include
            'robust' to use a robust loss function and have the model learn to predict an aleatoric
            uncertainty.
        df_or_path (str): Path to a data file to load with pandas.read_json().
        timestamp (str): Will prefix the names of model checkpoint files and other output files.
        folds (tuple[int, int] | None): If not None, split the data into n_folds[0] folds and use
            fold with index n_folds[1] as the test set. E.g. (10, 0) will create a 90/10 split
            and use first 10% as the test set.
        kwargs: Additional keyword arguments are passed to train_wrenformer().

    Raises:
        ValueError: On unknown dataset_name or invalid checkpoint.

    Returns:
        dict[str, float]: The model's test set metrics.
    """
    run_name = f"{model_name}-mp-{target_col}"

    if isinstance(df_or_path, str):
        df = pd.read_json(df_or_path).set_index(id_col, drop=False)
    else:
        df = df_or_path
    # shuffle samples for random train/test split
    df = df.sample(frac=1, random_state=0)

    if folds == test_size is None:
        raise ValueError(f"Must specify either {folds=} or {test_size=}")
    if folds is not None is not test_size:
        raise ValueError(f"Must specify either {folds=} or {test_size=}, not both")

    if folds is not None:
        n_folds, test_fold_idx = folds
        assert 1 < n_folds <= 10, f"{n_folds = } must be between 2 and 10"
        assert (
            0 <= test_fold_idx < n_folds
        ), f"{test_fold_idx = } must be between 0 and {n_folds - 1}"

        df_splits: list[pd.DataFrame] = np.array_split(df, n_folds)
        test_df = df_splits.pop(test_fold_idx)
        train_df = pd.concat(df_splits)
    if test_size is not None:
        assert 0 < test_size < 1, f"{test_size = } must be between 0 and 1"

        train_df = df.sample(frac=1 - test_size, random_state=0)
        test_df = df.drop(train_df.index)

    run_params = {"df_info": f"shape = {df.shape}, columns = {', '.join(df)}"}
    if isinstance(df_or_path, str):
        run_params["data_path"] = df_or_path

    test_metrics, *_ = train_wrenformer(
        run_name=run_name,
        train_df=train_df,
        test_df=test_df,
        target_col=target_col,
        task_type="regression",
        # set to None to disable logging
        wandb_project=kwargs.pop("wandb_project", "mp"),
        id_col=id_col,
        run_params=run_params,
        **kwargs,
    )

    return test_metrics


if __name__ == "__main__":
    timestamp = f"{datetime.now():%Y-%m-%d@%H-%M}"
    df = pd.read_json(f"{ROOT}/datasets/2022-08-13-mp-all-energies.json.gz")

    # for testing and debugging
    test_metrics = train_wrenformer_on_mp(
        model_name="wrenformer-robust-tmp",
        df_or_path=df,
        target_col="formation_energy_per_atom",
        folds=(10, 0),
        timestamp=timestamp,
        epochs=3,
        n_attn_layers=4,
        checkpoint=None,
        swa_start=None,
        verbose=True,
        # wandb_project=None,
    )
    print(f"{test_metrics=}")
