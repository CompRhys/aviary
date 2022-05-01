# %%
from __future__ import annotations

import os
from os.path import dirname, isfile

import pandas as pd
import torch
from matbench import MatbenchBenchmark
from matbench.task import MatbenchTask
from torch import nn

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


@print_walltime
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
        from aviary.wrenformer.data import collate_batch, get_initial_wyckoff_embedding
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

    if matbench_task.all_folds_recorded:
        print(f"\nTask {dataset_name} already recorded! Skipping...\n")
        return None

    df = pd.read_json(DATA_PATHS[dataset_name])
    # disable=None means hide pbar in non-tty but show when running interactively
    df["features"] = [get_initial_wyckoff_embedding(wyk_str) for wyk_str in df.wyckoff]
    matbench_task.df = df.set_index("mbid", drop=False)

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

    for fold in matbench_task.folds:
        if matbench_task.is_recorded[fold]:
            print(f"{fold = } of {dataset_name} already recorded! Skipping...")
            continue

        fold_name = f"{model_name}-{dataset_name}-{fold=}"

        train_df = matbench_task.get_train_and_val_data(fold, as_type="df")
        test_df = matbench_task.get_test_data(fold, as_type="df", include_target=True)

        train_loader = InMemoryDataLoader(
            [tuple(train_df[x]) for x in ["features", target, "mbid"]],
            batch_size=32,
            collate_fn=collate_batch,
        )
        test_loader = InMemoryDataLoader(
            [tuple(test_df[x]) for x in ["features", target, "mbid"]],
            batch_size=32,
            collate_fn=collate_batch,
        )

        # n_features = element + wyckoff embedding lengths + element weights in composition
        model = ModelClass(
            n_targets=[1], n_features=200 + 444 + 1, task_dict=task_dict, robust=robust
        )
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, verbose=True
        )

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
        )

        targets, [predictions], *ids = model.predict(test_loader)

        # record model predictions
        matbench_task.record(fold, predictions.cpu())

        # save model benchmark
        if isfile(benchmark_path):  # we checked for isfile() above but possible another
            # slurm job created it in the meantime in which case we merge results
            mbbm = MatbenchBenchmark.from_file(benchmark_path)
            mbbm.tasks_map[dataset_name] = matbench_task
        elif benchmark_dir := dirname(benchmark_path):
            os.makedirs(benchmark_dir, exist_ok=True)

        mbbm.to_file(benchmark_path)

    return mbbm


if __name__ == "__main__":
    # for testing and debugging
    run_matbench_task("wrenformer", "wrenformer-tmp.json", "matbench_jdft2d", 1)
