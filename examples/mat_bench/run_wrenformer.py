# %%
from __future__ import annotations

import os
from datetime import datetime
from typing import Literal

import pandas as pd
from matbench.task import MatbenchTask

from aviary.core import TaskType
from aviary.wrenformer.run import run_wrenformer
from aviary.wrenformer.utils import merge_json_on_disk
from examples.mat_bench import DATA_PATHS, MODULE_DIR


def run_wrenformer_on_matbench(
    model_name: str,
    dataset_name: str,
    fold: Literal[0, 1, 2, 3, 4],
    **kwargs,
) -> None:
    """Run a single matbench task.

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
        kwargs: Additional keyword arguments are passed to run_wrenformer(). See its doc string.

    Raises:
        ValueError: On unknown dataset_name or invalid checkpoint.

    Returns:
        dict[str, dict[str, list[float]]]: The model's test set metrics.
    """
    run_name = f"{model_name}-{dataset_name}-{fold=}"

    df = pd.read_json(DATA_PATHS[dataset_name]).set_index("mbid", drop=False)

    matbench_task = MatbenchTask(dataset_name, autoload=False)
    matbench_task.df = df

    target_name = matbench_task.metadata.target
    task_type: TaskType = matbench_task.metadata.task_type

    train_df = matbench_task.get_train_and_val_data(fold, as_type="df")
    test_df = matbench_task.get_test_data(fold, as_type="df", include_target=True)

    test_metrics, run_params = run_wrenformer(
        run_name=run_name,
        train_df=train_df,
        test_df=test_df,
        target_col=target_name,
        task_type=task_type,
        wandb_project="matbench",
        id_col=(id_col := "mbid"),
        run_params={
            "dataset": dataset_name,
            "fold": fold,
        },
        **kwargs,
    )

    # save model predictions to JSON
    preds_path = f"{MODULE_DIR}/model_preds/{model_name}-{timestamp}.json"

    # record model predictions
    preds_dict = test_df[[id_col, target_name, "predictions"]].to_dict(orient="list")
    merge_json_on_disk({dataset_name: {f"fold_{fold}": preds_dict}}, preds_path)

    # save model scores to JSON
    scores_path = f"{MODULE_DIR}/model_scores/{model_name}-{timestamp}.json"
    scores_dict = {dataset_name: {f"fold_{fold}": test_metrics}}
    scores_dict["params"] = run_params
    merge_json_on_disk(scores_dict, scores_path)

    print(f"scores for {fold = } of {dataset_name} written to {scores_path}")


# %%
if __name__ == "__main__":
    from glob import glob

    timestamp = f"{datetime.now():%Y-%m-%d@%H-%M}"

    try:
        # for testing and debugging
        run_wrenformer_on_matbench(
            model_name="roostformer-robust-mean+std-aggregation-tmp",
            # dataset_name="matbench_expt_is_metal",
            dataset_name="matbench_jdft2d",
            timestamp=timestamp,
            fold=0,
            epochs=10,
            log_wandb=True,
            checkpoint=None,
        )
    finally:  # clean up
        for filename in glob("model_*/*former-*-tmp-*.json"):
            os.remove(filename)
