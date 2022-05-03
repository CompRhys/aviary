# %%
from __future__ import annotations

import os
from datetime import datetime
from os.path import dirname, isfile
from typing import Literal

import numpy as np
import pandas as pd
import torch
from matbench import MatbenchBenchmark
from matbench.task import MatbenchTask
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from aviary import ROOT
from aviary.core import Normalizer
from aviary.data import InMemoryDataLoader
from aviary.losses import RobustL1Loss
from aviary.utils import print_walltime
from examples.mat_bench import DATA_PATHS, MatbenchDatasets

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
    benchmark_path: str,
    dataset_name: MatbenchDatasets,
    fold: Literal[0, 1, 2, 3, 4],
    epochs: int = 100,
) -> MatbenchBenchmark:
    """Run a single matbench task.

    Args:
        model_name (str): Can be any string to describe particular Roost/Wren variants.
        benchmark_path (str, optional): File path where to save benchmark results.
        dataset_name (str): Name of a matbench dataset like 'matbench_dielectric',
            'matbench_perovskites', etc. Unused if benchmark_path points at an already
            existing file.
        epochs (int): How many epochs to train for in each CV fold.

    Raises:
        ValueError: If dataset_name or benchmark_path is invalid.

    Returns:
        MatbenchBenchmark: Benchmark results.
    """
    if "wrenformer" in model_name.lower():
        from aviary.wrenformer.data import (
            collate_batch,
            wyckoff_embedding_from_aflow_str,
        )
        from aviary.wrenformer.model import Wrenformer as ModelClass
    # TODO: make it work with Wren and Roost too at some point, currently the model and
    # data class kwargs are differently named
    elif "wren" in model_name.lower():
        from aviary.wren.data import collate_batch
        from aviary.wren.model import Wren as ModelClass
    elif "roost" in model_name.lower():
        from aviary.roost.data import collate_batch
        from aviary.roost.model import Roost as ModelClass
    else:
        raise ValueError(f"Unexpected {model_name = }")

    if not benchmark_path.endswith(".json"):
        raise ValueError(f"{benchmark_path = } must have .json extension")
    if isfile(benchmark_path):
        mbbm = MatbenchBenchmark.from_file(benchmark_path)
    else:
        mbbm = MatbenchBenchmark(subset=[dataset_name])

    matbench_task: MatbenchTask = mbbm.tasks_map[dataset_name]

    if matbench_task.is_recorded[fold]:
        print(f"{fold = } of {dataset_name} already recorded! Skipping...")
        return mbbm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pytorch running on {device=}")

    df = pd.read_json(DATA_PATHS[dataset_name]).set_index("mbid", drop=False)
    with print_walltime("Generating initial Wyckoff embedding"):
        df["features"] = df.wyckoff.map(wyckoff_embedding_from_aflow_str)
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

    print(f"{matbench_task=}")
    train_df = matbench_task.get_train_and_val_data(fold, as_type="df")
    test_df = matbench_task.get_test_data(fold, as_type="df", include_target=True)

    features, targets, ids = (train_df[x] for x in ["features", target, "mbid"])
    targets = torch.tensor(targets, device=device)
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
    features_arr = np.empty(len(features), dtype=object)
    for idx, tensor in enumerate(features):
        features_arr[idx] = tensor.to(device)

    test_loader = InMemoryDataLoader(
        [features_arr, targets, ids], batch_size=1024, collate_fn=collate_batch
    )

    # n_features = element + wyckoff embedding lengths + element weights in composition
    model = ModelClass(
        n_targets=[1], n_features=200 + 444 + 1, task_dict=task_dict, robust=robust
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

    _, [predictions], *_ = model.predict(test_loader)

    if isfile(benchmark_path):  # we checked for isfile() above but possible another
        # slurm job created a partial benchmark in the meantime in which case we merge results
        mbbm = MatbenchBenchmark.from_file(benchmark_path)
        matbench_task = mbbm.tasks_map[dataset_name]
        # task needs df to pass validation before fold can be recorded
        matbench_task.df = df
    elif benchmark_dir := dirname(benchmark_path):
        os.makedirs(benchmark_dir, exist_ok=True)

    # record model predictions
    matbench_task.record(fold, predictions.cpu())
    mbbm.tasks_map[dataset_name] = matbench_task

    # save model benchmark
    mbbm.to_file(benchmark_path)

    return mbbm


# %%
if __name__ == "__main__":
    # for testing and debugging
    model_name = "wrenformer"
    benchmark_path = "wrenformer-tmp.json"
    dataset = "matbench_jdft2d"
    # dataset = "matbench_mp_is_metal"
    run_matbench_task(model_name, benchmark_path, dataset, 4, epochs=1)
