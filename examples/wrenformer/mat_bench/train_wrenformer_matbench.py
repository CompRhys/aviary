from __future__ import annotations

import os
from datetime import datetime
from typing import Literal

import pandas as pd
from matbench.task import MatbenchTask

from aviary.core import TaskType
from aviary.wrenformer.train import train_wrenformer
from aviary.wrenformer.utils import merge_json_on_disk
from examples.wrenformer.mat_bench import DATA_PATHS

__author__ = "Janosh Riebesell"
__date__ = "2022-04-11"

MODULE_DIR = os.path.dirname(__file__)


def train_wrenformer_on_matbench(
    model_name: str,
    dataset_name: str,
    fold: Literal[0, 1, 2, 3, 4],
    timestamp: str,
    **kwargs,
) -> dict[str, float]:
    """Train a Wrenformer on a single Matbench task.

    Args:
        model_name (str): Can be any string to describe particular Roost/Wren variants. Include
            'robust' to use a robust loss function and have the model learn to predict an aleatoric
            uncertainty.
        dataset_name (str): Name of a matbench dataset like 'matbench_dielectric',
            'matbench_perovskites', etc.
        fold (int): Which 5 CV folds of the dataset to use.
        timestamp (str): Timestamp to append to the names of JSON files for model predictions
            and performance scores. If the files already exist, results from different datasets
            or folds will be merged in.
        kwargs: Additional keyword arguments are passed to train_wrenformer().

    Raises:
        ValueError: On unknown dataset_name or invalid checkpoint.

    Returns:
        dict[str, float]: The model's test set metrics.
    """
    run_name = f"{model_name}-{dataset_name}-{fold=}"

    id_col = "mbid"
    df = pd.read_json(DATA_PATHS[dataset_name]).set_index(id_col, drop=False)

    matbench_task = MatbenchTask(dataset_name, autoload=False)
    matbench_task.df = df

    target = matbench_task.metadata.target
    task_type: TaskType = matbench_task.metadata.task_type

    train_df = matbench_task.get_train_and_val_data(fold, as_type="df")
    test_df = matbench_task.get_test_data(fold, as_type="df", include_target=True)

    test_metrics, run_params, test_df = train_wrenformer(
        run_name=run_name,
        train_df=train_df,
        test_df=test_df,
        target_col=target,
        task_type=task_type,
        # set to None to disable logging
        wandb_path=kwargs.pop("aviary/wandb_path", "mp-wbm"),
        id_col=id_col,
        run_params={
            "dataset": dataset_name,
            "fold": fold,
        },
        **kwargs,
    )

    # save model predictions to JSON
    preds_path = f"{MODULE_DIR}/model_preds/{timestamp}-{model_name}.json"

    # record model predictions
    preds_dict = test_df[[id_col, target, f"{target}_pred"]].to_dict(orient="list")
    merge_json_on_disk({dataset_name: {f"fold_{fold}": preds_dict}}, preds_path)

    # save model scores to JSON
    scores_path = f"{MODULE_DIR}/model_scores/{timestamp}-{model_name}.json"
    scores_dict = {dataset_name: {f"fold_{fold}": test_metrics}}
    scores_dict["params"] = run_params
    merge_json_on_disk(scores_dict, scores_path)

    print(f"scores for {fold = } of task {dataset_name} written to {scores_path}")

    return test_metrics


if __name__ == "__main__":
    from glob import glob

    timestamp = f"{datetime.now():%Y-%m-%d@%H-%M}"
    # dataset_name = "matbench_expt_is_metal"
    dataset_name = "matbench_jdft2d"

    try:
        # for testing and debugging
        test_metrics = train_wrenformer_on_matbench(
            model_name=f"roostformer-{dataset_name}-tmp",
            dataset_name=dataset_name,
            timestamp=timestamp,
            fold=0,
            epochs=25,
            wandb_path=None,
            verbose=True,
        )
        print(f"{test_metrics = }")
    finally:  # clean up
        for filename in glob("model_*/*former-*-tmp-*.json"):
            os.remove(filename)
