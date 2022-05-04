# %%
from __future__ import annotations

import gzip
import json
import os
from datetime import datetime
from os.path import dirname, isfile
from typing import Literal

import numpy as np
import pandas as pd
import torch
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
    n_transformer_layers: int = 4,
) -> dict[str, dict[str, list[float]]]:
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
        dict[str, dict[str, list[float]]]: Dictionary mapping {dataset_name: {fold: preds}}
            to model predictions.
    """
    if not benchmark_path.endswith(".json.gz"):
        raise ValueError(f"{benchmark_path = } must have .json.gz extension")
    if isfile(benchmark_path):
        with gzip.open(benchmark_path) as file:
            bench_dict = json.loads(file.read())
    else:
        bench_dict = {dataset_name: {}}

    if dataset_name in bench_dict and str(fold) in bench_dict[dataset_name]:
        print(f"{fold = } of {dataset_name} already recorded! Skipping...")
        return bench_dict

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
    assert n_features in (199 + 1, 200 + 1 + 444)  # Roost and Wren input dims resp.
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

    _, [predictions], *_ = model.predict(test_loader)

    if isfile(benchmark_path):  # we checked for isfile() above but possible another
        # slurm job created a partial benchmark in the meantime in which case we merge results
        with gzip.open(benchmark_path) as file:
            bench_dict = json.load(file)
    elif benchmark_dir := dirname(benchmark_path):
        os.makedirs(benchmark_dir, exist_ok=True)

    if dataset_name not in bench_dict:
        bench_dict[dataset_name] = {}
    # record model predictions
    bench_dict[dataset_name][fold] = predictions.cpu().tolist()

    # save model benchmark
    with gzip.open(benchmark_path, "w") as file:
        file.write(json.dumps(bench_dict).encode("utf-8"))
    print(f"{fold = } of {dataset_name} written to {benchmark_path=}")

    return bench_dict


# %%
if __name__ == "__main__":
    # for testing and debugging
    model_name = "roost"
    benchmark_path = "benchmarks/wrenformer-tmp.json.gz"
    dataset = "matbench_jdft2d"
    # dataset = "matbench_mp_is_metal"
    run_matbench_task(model_name, benchmark_path, dataset, 4, epochs=1)
    os.remove(benchmark_path)
