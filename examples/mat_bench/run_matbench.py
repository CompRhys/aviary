# %%
from __future__ import annotations

import os
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import torch
import wandb
from matbench.constants import CLF_KEY, REG_KEY
from matbench.task import MatbenchTask
from torch import nn

from aviary.core import Normalizer
from aviary.data import InMemoryDataLoader
from aviary.losses import RobustL1Loss
from aviary.utils import get_metrics
from aviary.wrenformer.data import (
    collate_batch,
    get_composition_embedding,
    wyckoff_embedding_from_aflow_str,
)
from aviary.wrenformer.model import Wrenformer
from examples.mat_bench import DATA_PATHS, MODULE_DIR, MatbenchDatasets
from examples.mat_bench.utils import merge_json_on_disk, print_walltime

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
    log_wandb: bool = True,
) -> dict:
    """Run a single matbench task.

    Args:
        model_name (str): Can be any string to describe particular Roost/Wren variants.
        dataset_name (str): Name of a matbench dataset like 'matbench_dielectric',
            'matbench_perovskites', etc.
        timestamp (str): Timestamp to append to the names of JSON files for model predictions
            and performance scores. If the files already exist, results from different datasets
            or folds will be merged in.
        epochs (int): How many epochs to train for in each CV fold.
        n_transformer_layers (int): Number of transformer layers to use. Default is 4.
        wandb (bool): Whether to log this run to Weights and Biases. Defaults to True.

    Raises:
        ValueError: On unknown dataset_name.

    Returns:
        dict[str, dict[str, list[float]]]: Dictionary mapping {dataset_name: {fold: preds}}
            to model predictions.
    """
    scores_path = f"{MODULE_DIR}/model_scores/{model_name}-{timestamp}.json"

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

    matbench_task = MatbenchTask(dataset_name, autoload=False)
    matbench_task.df = df

    target, task_type = (
        str(matbench_task.metadata[x]) for x in ("target", "task_type")
    )

    robust = False
    loss_func = (
        (RobustL1Loss if robust else nn.L1Loss())
        if task_type == REG_KEY
        else (nn.NLLLoss() if robust else nn.CrossEntropyLoss())
    )
    loss_dict = {target: (task_type, loss_func)}
    normalizer_dict = {target: Normalizer() if task_type == REG_KEY else None}

    fold_name = f"{model_name}-{dataset_name}-{fold=}"

    train_df = matbench_task.get_train_and_val_data(fold, as_type="df")
    test_df = matbench_task.get_test_data(fold, as_type="df", include_target=True)

    features, targets, ids = (train_df[x] for x in ["features", target, "mbid"])
    targets = torch.tensor(targets, device=device)
    if targets.dtype == torch.bool:
        targets = targets.long()
    inputs = np.empty(len(features), dtype=object)
    for idx, tensor in enumerate(features):
        inputs[idx] = tensor.to(device)

    train_loader = InMemoryDataLoader(
        [inputs, targets, ids], batch_size=32, shuffle=True, collate_fn=collate_batch
    )

    features, targets, ids = (test_df[x] for x in ["features", target, "mbid"])
    targets = torch.tensor(targets, device=device)
    if targets.dtype == torch.bool:
        targets = targets.long()
    inputs = np.empty(len(features), dtype=object)
    for idx, tensor in enumerate(features):
        inputs[idx] = tensor.to(device)

    test_loader = InMemoryDataLoader(
        [inputs, targets, ids], batch_size=128, collate_fn=collate_batch
    )

    n_features = features[0].shape[-1]
    assert n_features in (200 + 1, 200 + 1 + 444)  # Roost and Wren embedding size resp.

    # n_features = element + wyckoff embedding lengths + element weights in composition
    model = Wrenformer(
        n_targets=[1 if task_type == REG_KEY else 2],
        n_features=n_features,
        task_dict={target: task_type},  # e.g. {'exfoliation_en': 'regression'}
        n_transformer_layers=n_transformer_layers,
        robust=robust,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, verbose=True)

    if log_wandb:
        wandb.login()
        wandb.init(
            project="matbench",  # run will be added to this project
            # https://docs.wandb.ai/guides/track/launch#init-start-error
            settings=wandb.Settings(start_method="fork"),
            name=fold_name,
            config={
                "model": model_name,
                "dataset": dataset_name,
                "fold": fold,
                "epochs": epochs,
                "n_transformer_layers": n_transformer_layers,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
            },
        )

    model.fit(
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        epochs=epochs,
        loss_dict=loss_dict,
        normalizer_dict=normalizer_dict,
        model_name=fold_name,
        run_id=1,
        checkpoint=False,
        writer="wandb" if log_wandb else None,
        verbose=False,
    )

    _, [predictions], _ = model.predict(test_loader)
    if task_type == CLF_KEY:
        predictions = predictions.softmax(dim=1)
    predictions = predictions.cpu().numpy().squeeze()
    targets = targets.cpu().numpy()
    test_df["predictions"] = predictions.tolist()

    metrics = get_metrics(targets, predictions, task_type)

    if log_wandb:
        wandb.summary = {"test": metrics}
        if task_type == REG_KEY:
            import plotly.express as px
            from pymatviz.utils import add_identity_line

            fig = px.scatter(
                test_df,
                x=target,
                y="predictions",
                hover_data=["mbid"],
                opacity=0.7,
                width=1200,
                height=800,
            )
            add_identity_line(fig)
            fig.update_yaxes(title=f"predicted {target}")

            plots = {"scatter": fig}
        elif task_type == CLF_KEY:
            from examples.mat_bench.plotting_functions import plotly_roc

            fig = plotly_roc(targets, predictions[:, 1])
            plots = {"roc": fig}

        wandb.log(plots)
        wandb.finish()

    # save model predictions to JSON
    preds_path = f"{MODULE_DIR}/model_preds/{model_name}-{timestamp}.json"
    params = {
        "epochs": epochs,
        "n_transformer_layers": n_transformer_layers,
        "learning_rate": learning_rate,
        "robust": robust,
        "n_features": n_features,  # embedding size
        dataset_name: {"losses": str(loss_dict)},
    }

    # record model predictions
    preds_dict = test_df[["mbid", target, "predictions"]].to_dict(orient="list")
    merge_json_on_disk({dataset_name: {f"fold_{fold}": preds_dict}}, preds_path)

    # save model scores to JSON
    scores_dict = {dataset_name: {f"fold_{fold}": metrics}}
    scores_dict["params"] = params
    merge_json_on_disk(scores_dict, scores_path)

    print(f"scores for {fold = } of {dataset_name} written to {scores_path}")
    return scores_dict


# %%
if __name__ == "__main__":
    from glob import glob

    timestamp = f"{datetime.now():%Y-%m-%d@%H-%M}"

    try:
        # for testing and debugging
        run_matbench_task(
            model_name := "roostformer-tmp",
            # dataset_name="matbench_expt_is_metal",
            dataset_name="matbench_jdft2d",
            timestamp=timestamp,
            fold=3,
            epochs=1,
            log_wandb=True,
        )
    finally:  # clean up
        for filename in glob("model_*/*former-tmp-*.json"):
            os.remove(filename)
