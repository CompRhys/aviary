# %%
from __future__ import annotations

import os
from os.path import dirname, isfile

import pandas as pd
import torch
from matbench import MatbenchBenchmark
from torch import nn

from aviary.core import Normalizer
from aviary.losses import RobustL1Loss
from examples.mat_bench import DATA_PATHS, MatbenchDatasets

__author__ = "Janosh Riebesell"
__date__ = "2022-04-11"
# Related Matbench issue: https://github.com/materialsproject/matbench/issues/116

torch.manual_seed(0)  # ensure reproducible results


# %%
learning_rate = 1e-3
warmup_steps = 60


def lr_lambda(epoch: int) -> float:
    return min((epoch + 1) ** (-0.5), (epoch + 1) * warmup_steps ** (-1.5))


def run_matbench_task(
    model_name: str,
    benchmark_path: str,
    dataset_name: MatbenchDatasets,
    epochs: int = 100,
) -> MatbenchBenchmark | None:
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
        from aviary.wrenformer.data import WyckoffData as DataClass
        from aviary.wrenformer.data import collate_batch
        from aviary.wrenformer.model import Wren as ModelClass
    # TODO: make it work with Wren and Roost too at some point, currently the model and
    # data class kwargs are differently named
    elif "wren" in model_name.lower():
        from aviary.wren.data import WyckoffData as DataClass
        from aviary.wren.data import collate_batch
        from aviary.wren.model import Wren as ModelClass
    elif "roost" in model_name.lower():
        from aviary.roost.data import CompositionData as DataClass
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

    matbench_task = mbbm.tasks_map[dataset_name]

    if matbench_task.all_folds_recorded:
        print(f"\nTask {dataset_name} already recorded! Skipping...\n")
        return None

    df = pd.read_json(DATA_PATHS[dataset_name])

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

    matbench_task.df = df.set_index("mbid", drop=False)

    for fold in matbench_task.folds:
        if matbench_task.is_recorded[fold]:
            print(f"{fold = } of {dataset_name} already recorded! Skipping...")
            continue

        model_name = f"roost-{dataset_name}-{fold=}"

        train_df = matbench_task.get_train_and_val_data(fold, as_type="df")
        test_df = matbench_task.get_test_data(fold, as_type="df", include_target=True)

        train_set = DataClass(train_df, task_dict, id_cols=["mbid"])

        # n_features = element + wyckoff embedding lengths + element weights in composition
        model = ModelClass(
            robust=False, n_targets=[1], n_features=200 + 444 + 1, task_dict=task_dict
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, verbose=True
        )

        test_set = DataClass(test_df, task_dict, id_cols=["mbid"])
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=32, collate_fn=collate_batch
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=32, collate_fn=collate_batch
        )

        model.fit(
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            epochs=epochs,
            criterion_dict=loss_dict,
            normalizer_dict=normalizer_dict,
            model_name=model_name,
            run_id=1,
        )

        targets, [predictions], *ids = model.predict(test_loader)

        # record model predictions
        matbench_task.record(fold, predictions.cpu())

    # save model benchmark
    if isfile(benchmark_path):  # we checked for isfile() above but possible another
        # slurm job created it in the meantime in which case we need to merge results
        mbbm = MatbenchBenchmark.from_file(benchmark_path)
        mbbm.tasks_map[dataset_name] = matbench_task
    elif benchmark_dir := dirname(benchmark_path):
        os.makedirs(benchmark_dir, exist_ok=True)

    mbbm.to_file(benchmark_path)

    return mbbm
