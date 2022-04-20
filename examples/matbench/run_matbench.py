"""Author: Janosh Riebesell. Started 2022-04-11.
Related: https://github.com/materialsproject/matbench/issues/116
"""

# %%
from __future__ import annotations

import os
from datetime import datetime
from os.path import dirname, isfile
from typing import Any

import pandas as pd
import torch
from matbench import MatbenchBenchmark

from aviary.roost.model import Roost
from aviary.utils import results_multitask, train_ensemble
from aviary.wren.model import Wren
from examples.matbench import DATA_PATHS, MatbenchDatasets

# %%
torch.manual_seed(0)  # ensure reproducible results

resume = False
fine_tune = None
transfer = None


device = "cuda" if torch.cuda.is_available() else "cpu"
identifiers = ("mbid", "composition")
robust = True
elem_emb = "matscholar200"

ensemble = 1
run_id = 1
log = True

data_params: dict[str, Any] = {
    "batch_size": 128,
    "num_workers": 0,
    "pin_memory": False,
    "shuffle": True,
}
setup_params = {
    "optim": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-6,
    "momentum": 0.9,
    "device": device,
}
restart_params = {
    "resume": resume,
    "fine_tune": fine_tune,
    "transfer": transfer,
}


# %%
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
    if "roost" in model_name.lower():
        model_cls = Roost
    elif "wren" in model_name.lower():
        model_cls = Wren
    else:
        raise ValueError(f"{model_name = } must contain either 'wren' or 'roost'")

    if not benchmark_path.endswith(".json"):
        raise ValueError(f"{benchmark_path = } must have .json extension")
    if isfile(benchmark_path):
        mbbm = MatbenchBenchmark.from_file(benchmark_path)
    else:
        mbbm = MatbenchBenchmark(subset=[dataset_name])

    task = mbbm.tasks_map[dataset_name]

    if task.all_folds_recorded:
        print(f"\nTask {dataset_name} already recorded! Skipping...\n")
        return None

    df = pd.read_json(DATA_PATHS[dataset_name])

    target, task_type = (str(task.metadata[x]) for x in ("target", "task_type"))
    task_dict = {target: task_type}
    loss_dict = {target: "L1" if task_type == "regression" else "CSE"}

    if model_cls == Roost:
        from aviary.roost.data import CompositionData as data_cls
        from aviary.roost.data import collate_batch
    else:
        from aviary.wren.data import WyckoffData as data_cls
        from aviary.wren.data import collate_batch
        from aviary.wren.utils import count_wyks

    task.df = df.set_index("mbid", drop=False)
    data_params["collate_fn"] = collate_batch

    for fold in task.folds:
        if task.is_recorded[fold]:
            print(f"{fold = } of {dataset_name} already recorded! Skipping...")
            continue

        model_name = f"roost-{dataset_name}-{fold=}"

        train_df = task.get_train_and_val_data(fold, as_type="df")
        test_df = task.get_test_data(fold, as_type="df", include_target=True)

        # need to restrict max number of Wyckoff positions to avoid OOM errors
        # (even on 80 GB A100 GPUs)
        if model_cls == Wren:
            train_df = train_df[[count_wyks(x) < 20 for x in train_df.wyckoff]]

        train_set = data_cls(train_df, task_dict, elem_emb, identifiers=identifiers)

        test_set = data_cls(test_df, task_dict, elem_emb, identifiers=identifiers)

        model_params = {
            "task_dict": task_dict,
            "robust": robust,
            "n_targets": train_set.n_targets,
            "elem_emb_len": train_set.elem_emb_len,
            "elem_fea_len": 64,
            "n_graph": 3,
            "elem_heads": 3,
            "elem_gate": [256],
            "elem_msg": [256],
            "cry_heads": 3,
            "cry_gate": [256],
            "cry_msg": [256],
            "trunk_hidden": [128, 128],
            "out_hidden": [64, 64],
        }

        if model_cls == Wren:
            model_params["sym_emb_len"] = train_set.sym_emb_len

        train_ensemble(
            model_class=model_cls,
            model_name=model_name,
            run_id=run_id,
            ensemble_folds=ensemble,
            epochs=epochs,
            train_set=train_set,
            val_set=test_set,
            log=log,
            data_params=data_params,
            setup_params=setup_params,
            restart_params=restart_params,
            model_params=model_params,
            loss_dict=loss_dict,
        )

        results_dict = results_multitask(
            model_class=model_cls,
            model_name=model_name,
            run_id=run_id,
            ensemble_folds=ensemble,
            test_set=test_set,
            # we set shuffle=False since ensembling needs fixed data order
            data_params={**data_params, "shuffle": False},
            robust=robust,
            task_dict=task_dict,
            device=device,
            eval_type="checkpoint",
            save_results=False,
        )

        predictions = results_dict[target]["pred"].ravel()

        # record model predictions
        task.record(fold, predictions)

    # save model benchmark
    if isfile(benchmark_path):  # we checked for isfile() above but possible another
        # slurm job created it in the meantime in which case we need to merge results
        mbbm = MatbenchBenchmark.from_file(benchmark_path)
        mbbm.tasks_map[dataset_name] = task
    else:
        os.makedirs(dirname(benchmark_path), exist_ok=True)

    mbbm.to_file(benchmark_path)

    return mbbm


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model", choices=["roost", "wren"])
    args = parser.parse_args()

    mbbm = MatbenchBenchmark()
    benchmark_path = f"{args.model}-matbench-{datetime.now():%Y-%m-%d@%H-%M}.json"

    model_cls = Roost if args.model == "roost" else Wren

    for idx, dataset_name in enumerate(DATA_PATHS, 1):
        print(f"\n\n{idx}/{len(DATA_PATHS)}")
        run_matbench_task(dataset_name, benchmark_path, args.model, model_cls)
