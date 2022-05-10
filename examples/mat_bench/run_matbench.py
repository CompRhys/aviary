# %%
from __future__ import annotations

import os
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import torch
from matbench.data_ops import score_array
from matbench.task import MatbenchTask
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from aviary import ROOT
from aviary.core import Normalizer
from aviary.data import InMemoryDataLoader
from aviary.losses import RobustL1Loss
from aviary.utils import print_walltime
from aviary.wrenformer.data import (
    collate_batch,
    get_composition_embedding,
    wyckoff_embedding_from_aflow_str,
)
from aviary.wrenformer.model import Wrenformer
from examples.mat_bench import DATA_PATHS, MODULE_DIR, MatbenchDatasets
from examples.mat_bench.utils import open_json

__author__ = "Janosh Riebesell"
__date__ = "2022-04-11"
# Related Matbench issue: https://github.com/materialsproject/matbench/issues/116

torch.manual_seed(0)  # ensure reproducible results


# %%
learning_rate = 1e-3
warmup_steps = 10


def lr_lambda(epoch: int) -> float:
    return min((epoch + 1) ** (-0.5), (epoch + 1) * warmup_steps ** (-1.5))


@print_walltime("run_matbench_task()")
def run_matbench_task(
    model_name: str,
    dataset_name: MatbenchDatasets,
    timestamp: str,
    fold: Literal[0, 1, 2, 3, 4],
    epochs: int = 100,
    n_transformer_layers: int = 4,
) -> dict[str, dict[str, list[float]]]:
    """Run a single matbench task.

    Args:
        model_name (str): Can be any string to describe particular Roost/Wren variants.
        dataset_name (str): Name of a matbench dataset like 'matbench_dielectric',
            'matbench_perovskites', etc.
        timestamp (str): Timestamp to append to the names of JSON files for model predictions
            and performance scores. If the files already exist, results from different datasets
            or folds will be merged in.
        epochs (int): How many epochs to train for in each CV fold.

    Raises:
        ValueError: On unknown dataset_name.

    Returns:
        dict[str, dict[str, list[float]]]: Dictionary mapping {dataset_name: {fold: preds}}
            to model predictions.
    """
    benchmark_path = f"{MODULE_DIR}/benchmarks/preds-{model_name}-{timestamp}.json.gz"
    scores_path = f"{MODULE_DIR}/benchmarks/scores-{model_name}-{timestamp}.json"

    with open_json(scores_path) as json_data:
        scores_dict = json_data

    if dataset_name in scores_dict and str(fold) in scores_dict[dataset_name]:
        print(f"{fold = } of {dataset_name} already recorded! Skipping...")
        return scores_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pytorch running on {device=}")

    df = pd.read_json(DATA_PATHS[dataset_name]).set_index("mbid", drop=False)
    if "wren" in model_name.lower():
        with print_walltime("Generating initial Wyckoff embedding"):
            df["features"] = df.wyckoff.map(wyckoff_embedding_from_aflow_str)
    elif "roost" in model_name.lower():
        df["features"] = df.composition.map(get_composition_embedding)
    else:
        raise ValueError(f"{model_name = } must contain 'roost' or 'wren'")

    n_features = df.features.iloc[0].shape[1]
    assert n_features in (199 + 1, 200 + 1 + 444)  # Roost and Wren embedding size resp.
    matbench_task = MatbenchTask(dataset_name, autoload=False)
    matbench_task.df = df

    target, task_type = (
        str(matbench_task.metadata[x]) for x in ("target", "task_type")
    )
    task_dict = {target: task_type}  # e.g. {'exfoliation_en': 'regression'}

    robust = False
    loss_func = (
        (RobustL1Loss if robust else nn.L1Loss())
        if task_type == "regression"
        else (nn.NLLLoss() if robust else nn.CrossEntropyLoss())
    )
    loss_dict = {target: (task_type, loss_func)}
    normalizer_dict = {target: Normalizer() if task_type == "regression" else None}

    fold_name = f"{model_name}-{dataset_name}-{fold=}"

    train_df = matbench_task.get_train_and_val_data(fold, as_type="df")
    test_df = matbench_task.get_test_data(fold, as_type="df", include_target=True)

    features, targets, ids = (train_df[x] for x in ["features", target, "mbid"])
    targets = torch.tensor(targets, device=device)
    if targets.dtype == torch.bool:
        targets = targets.long()
    features_arr = np.empty(len(features), dtype=object)
    for idx, tensor in enumerate(features):
        features_arr[idx] = tensor.to(device)

    train_loader = InMemoryDataLoader(
        [features_arr, targets, ids],
        batch_size=32,
        shuffle=True,
        collate_fn=collate_batch,
    )

    features, targets, ids = (test_df[x] for x in ["features", target, "mbid"])
    targets = torch.tensor(targets, device=device)
    if targets.dtype == torch.bool:
        targets = targets.long()
    features_arr = np.empty(len(features), dtype=object)
    for idx, tensor in enumerate(features):
        features_arr[idx] = tensor.to(device)

    test_loader = InMemoryDataLoader(
        [features_arr, targets, ids], batch_size=128, collate_fn=collate_batch
    )

    # n_features = element + wyckoff embedding lengths + element weights in composition
    model = Wrenformer(
        n_targets=[1 if task_type == "regression" else 2],
        n_features=n_features,
        task_dict=task_dict,
        n_transformer_layers=n_transformer_layers,
        robust=robust,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, verbose=True)

    writer = SummaryWriter(f"{ROOT}/runs/{fold_name}/{datetime.now():%Y-%m-%d@%H-%M}")

    model.fit(
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        epochs=epochs,
        criterion_dict=loss_dict,
        normalizer_dict=normalizer_dict,
        model_name=fold_name,
        run_id=1,
        checkpoint=False,
        writer=writer,
    )

    [targets], [predictions], *ids = model.predict(test_loader)
    predictions = predictions.cpu().squeeze()

    params = {
        "epochs": epochs,
        "n_transformer_layers": n_transformer_layers,
        "learning_rate": learning_rate,
        "robust": robust,
        "n_features": n_features,  # embedding size
        "losses": str(loss_dict),
    }

    # --- save model predictions to gzipped JSON ---
    with open_json(benchmark_path) as bench_dict:
        # record model predictions
        bench_dict[dataset_name][fold] = {
            "preds": predictions.tolist(),
            "ids": list(ids),
            "params": params,
        }

    # --- save model scores to JSON ---
    # matbench.data_ops.score_array() calculates calculates [MAE, RMSE, MAPE, max error]
    # for regression and [accuracy, balanced accuracy, F1, ROCAUC] for classification.
    scores = score_array(predictions, targets, task_type)

    with open_json(scores_path) as scores_dict:
        scores_dict[dataset_name][fold] = scores
        scores_dict["params"] = params

    print(f"scores for {fold = } of {dataset_name} written to {scores_path}")
    return bench_dict


# %%
if __name__ == "__main__":
    from glob import glob

    try:
        # for testing and debugging
        model_name = "roostformer-tmp"
        timestamp = f"{datetime.now():%Y-%m-%d@%H-%M}"
        dataset = "matbench_jdft2d"
        # dataset = "matbench_mp_is_metal"
        run_matbench_task(model_name, dataset, timestamp, 4, epochs=1)
    finally:  # clean up
        for filename in glob(f"benchmarks/*{model_name}-{timestamp}*.json*"):
            os.remove(filename)
