# %%
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from aviary import ROOT
from examples.wrenformer import run_wrenformer

__author__ = "Janosh Riebesell"
__date__ = "2022-06-13"


def run_wrenformer_on_mp_wbm(
    model_name: str,
    data_path: str,
    target: str,
    folds: tuple[int, int] | None = None,
    test_size: float | None = None,
    wandb_project: str | None = "mp-wbm",
    **kwargs,
) -> dict[str, float]:
    """Run a single matbench task.

    Args:
        model_name (str): Can be any string to describe particular Roost/Wren variants. Include
            'robust' to use a robust loss function and have the model learn to predict an aleatoric
            uncertainty.
        data_path (str): Path to a data file to load with pandas.read_json().
        timestamp (str): Will prefix the names of model checkpoint files and other output files.
        wandb_project (str | None): Project name to use when logging to wandb. Defaults to "mp-wbm".
            Set to None to disable logging.
        kwargs: Additional keyword arguments are passed to run_wrenformer(). See its doc string.

    Raises:
        ValueError: On unknown dataset_name or invalid checkpoint.

    Returns:
        dict[str, float]: The model's test set metrics.
    """
    run_name = f"{model_name}-mp+wbm-{target}"

    id_col = "material_id"
    df = pd.read_json(data_path).set_index(id_col, drop=False)
    df = df.sample(frac=1)  # shuffle samples

    if (folds, test_size) == (None, None):
        raise ValueError("Must specify either folds or test_size")
    if folds is not None and test_size is not None:
        raise ValueError("Must specify either folds or test_size, not both")

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

    test_metrics, *_ = run_wrenformer(
        run_name=run_name,
        train_df=train_df,
        test_df=test_df,
        target_col=target,
        task_type="regression",
        wandb_project=wandb_project,
        id_col=id_col,
        run_params={
            "data_path": data_path,
        },
        **kwargs,
    )

    return test_metrics


# %%
if __name__ == "__main__":
    timestamp = f"{datetime.now():%Y-%m-%d@%H-%M}"

    # for testing and debugging
    test_metrics = run_wrenformer_on_mp_wbm(
        model_name="wrenformer-robust-tmp",
        target="e_form",
        data_path=f"{ROOT}/datasets/2022-06-09-mp+wbm.json.gz",
        fold=0,
        timestamp=timestamp,
        epochs=3,
        n_attn_layers=4,
        checkpoint=None,
        swa=True,
        wandb_project=None,
    )
    print(f"{test_metrics=}")
