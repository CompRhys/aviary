import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split as split

from aviary.utils import get_metrics, results_multitask, train_ensemble
from aviary.wrenformer.data import df_to_in_mem_dataloader
from aviary.wrenformer.model import Wrenformer


@pytest.fixture
def base_config():
    return {
        "robust": True,
        "ensemble": 2,
        "run_id": 1,
        "data_seed": 42,
        "log": False,
        "sample": 1,
        "test_size": 0.2,
    }


@pytest.fixture
def model_architecture():
    return {
        "d_model": 128,
        "n_attn_layers": 6,
        "n_attn_heads": 4,
        "trunk_hidden": (1024, 512),
        "out_hidden": (256, 128, 64),
        "embedding_aggregations": ("mean",),
    }


@pytest.fixture
def training_config():
    return {
        "resume": False,
        "fine_tune": None,
        "transfer": None,
        "optim": "AdamW",
        "learning_rate": 3e-4,
        "momentum": 0.9,
        "weight_decay": 1e-6,
        "batch_size": 128,
        "workers": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def test_wrenformer_regression(
    df_matbench_phonons_wyckoff, base_config, model_architecture, training_config
):
    target_name = "last phdos peak"
    task = "regression"
    losses = ["L1"]
    epochs = 25
    model_name = "wrenformer-reg-test"
    input_col = "wyckoff"
    embedding_type = "wyckoff"

    task_dict = dict(zip([target_name], [task], strict=False))
    loss_dict = dict(zip([target_name], losses, strict=False))

    train_idx = list(range(len(df_matbench_phonons_wyckoff)))
    train_idx, test_idx = split(
        train_idx,
        random_state=base_config["data_seed"],
        test_size=base_config["test_size"],
    )

    train_df = df_matbench_phonons_wyckoff.iloc[train_idx[0 :: base_config["sample"]]]
    test_df = df_matbench_phonons_wyckoff.iloc[test_idx]
    val_df = test_df  # Using test set for validation

    data_loader_kwargs = dict(
        id_col="material_id",
        input_col=input_col,
        target_col=target_name,
        embedding_type=embedding_type,
        device=training_config["device"],
    )

    train_loader = df_to_in_mem_dataloader(
        train_df,
        batch_size=training_config["batch_size"],
        shuffle=True,
        **data_loader_kwargs,
    )

    val_loader = df_to_in_mem_dataloader(
        val_df,
        batch_size=training_config["batch_size"] * 16,
        shuffle=False,
        **data_loader_kwargs,
    )

    setup_params = {
        "optim": training_config["optim"],
        "learning_rate": training_config["learning_rate"],
        "weight_decay": training_config["weight_decay"],
        "momentum": training_config["momentum"],
        "device": training_config["device"],
    }

    restart_params = {
        "resume": training_config["resume"],
        "fine_tune": training_config["fine_tune"],
        "transfer": training_config["transfer"],
    }

    n_targets = [1]  # Regression task has 1 target

    model_params = {
        "task_dict": task_dict,
        "robust": base_config["robust"],
        "n_targets": n_targets,
        "n_features": train_loader.tensors[0][0].shape[-1],
        **model_architecture,
    }

    train_ensemble(
        model_class=Wrenformer,
        model_name=model_name,
        run_id=base_config["run_id"],
        ensemble_folds=base_config["ensemble"],
        epochs=epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        log=base_config["log"],
        setup_params=setup_params,
        restart_params=restart_params,
        model_params=model_params,
        loss_dict=loss_dict,
    )

    test_loader = df_to_in_mem_dataloader(
        test_df,
        batch_size=training_config["batch_size"] * 64,
        shuffle=False,
        **data_loader_kwargs,
    )

    results_dict = results_multitask(
        model_class=Wrenformer,
        model_name=model_name,
        run_id=base_config["run_id"],
        ensemble_folds=base_config["ensemble"],
        test_loader=test_loader,
        robust=base_config["robust"],
        task_dict=task_dict,
        device=training_config["device"],
        eval_type="checkpoint",
        save_results=False,
    )

    preds = results_dict[target_name]["preds"]
    targets = results_dict[target_name]["targets"]

    y_ens = np.mean(preds, axis=0)
    mae, rmse, r2 = get_metrics(targets, y_ens, task).values()

    assert len(targets) == len(test_df)
    assert r2 > 0.7
    assert mae < 150
    assert rmse < 300


if __name__ == "__main__":
    pytest.main(["-v", __file__])
