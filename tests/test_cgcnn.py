import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split as split
from torch.utils.data import DataLoader

from aviary.cgcnn.data import CrystalGraphData, collate_batch
from aviary.cgcnn.model import CrystalGraphConvNet
from aviary.utils import get_metrics, results_multitask, train_ensemble


@pytest.fixture
def base_config():
    return {
        "elem_embedding": "cgcnn92",
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
        "elem_fea_len": 32,
        "h_fea_len": 128,
        "n_graph": 3,
        "n_hidden": 1,
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


def test_cgcnn_regression(
    df_matbench_phonons, base_config, model_architecture, training_config
):
    target_name = "last phdos peak"
    task = "regression"
    losses = ["L1"]
    epochs = 25
    model_name = "cgcnn-reg-test"

    task_dict = dict(zip([target_name], [task], strict=False))
    loss_dict = dict(zip([target_name], losses, strict=False))

    dataset = CrystalGraphData(
        df=df_matbench_phonons,
        elem_embedding=base_config["elem_embedding"],
        task_dict=task_dict,
    )
    n_targets = dataset.n_targets
    elem_emb_len = dataset.elem_emb_len
    nbr_fea_len = dataset.nbr_fea_dim

    train_idx = list(range(len(dataset)))
    train_idx, test_idx = split(
        train_idx,
        random_state=base_config["data_seed"],
        test_size=base_config["test_size"],
    )
    test_set = torch.utils.data.Subset(dataset, test_idx)
    val_set = test_set
    train_set = torch.utils.data.Subset(dataset, train_idx[0 :: base_config["sample"]])

    data_params = {
        "batch_size": training_config["batch_size"],
        "num_workers": training_config["workers"],
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": collate_batch,
    }

    train_loader = DataLoader(train_set, **data_params)
    val_loader = DataLoader(
        val_set,
        **{**data_params, "batch_size": 16 * data_params["batch_size"], "shuffle": False},
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

    model_params = {
        "task_dict": task_dict,
        "robust": base_config["robust"],
        "n_targets": n_targets,
        "elem_emb_len": elem_emb_len,
        "nbr_fea_len": nbr_fea_len,
        **model_architecture,
    }

    train_ensemble(
        model_class=CrystalGraphConvNet,
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

    test_loader = DataLoader(
        test_set,
        **{**data_params, "batch_size": 64 * data_params["batch_size"], "shuffle": False},
    )

    results_dict = results_multitask(
        model_class=CrystalGraphConvNet,
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

    assert len(targets) == len(test_set) == len(test_idx)
    assert r2 > 0.7
    assert mae < 150
    assert rmse < 300


def test_cgcnn_clf(df_matbench_phonons, base_config, model_architecture, training_config):
    targets = ["phdos_clf"]
    task = "classification"
    losses = ["CSE"]
    epochs = 10
    model_name = "cgcnn-clf-test"

    task_dict = dict(zip(targets, [task], strict=False))
    loss_dict = dict(zip(targets, losses, strict=False))

    dataset = CrystalGraphData(
        df=df_matbench_phonons,
        elem_embedding=base_config["elem_embedding"],
        task_dict=task_dict,
    )
    n_targets = dataset.n_targets
    elem_emb_len = dataset.elem_emb_len
    nbr_fea_len = dataset.nbr_fea_dim

    train_idx = list(range(len(dataset)))

    print(f"using {base_config['test_size']} of training set as test set")
    train_idx, test_idx = split(
        train_idx,
        random_state=base_config["data_seed"],
        test_size=base_config["test_size"],
    )
    test_set = torch.utils.data.Subset(dataset, test_idx)

    print("No validation set used, using test set for evaluation purposes")
    # NOTE that when using this option care must be taken not to
    # peak at the test-set. The only valid model to use is the one
    # obtained after the final epoch where the epoch count is
    # decided in advance of the experiment.
    val_set = test_set

    train_set = torch.utils.data.Subset(dataset, train_idx[0 :: base_config["sample"]])

    data_params = {
        "batch_size": training_config["batch_size"],
        "num_workers": training_config["workers"],
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": collate_batch,
    }

    train_loader = DataLoader(train_set, **data_params)
    val_loader = DataLoader(
        val_set,
        **{**data_params, "batch_size": 16 * data_params["batch_size"], "shuffle": False},
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

    model_params = {
        "task_dict": task_dict,
        "robust": base_config["robust"],
        "n_targets": n_targets,
        "elem_emb_len": elem_emb_len,
        "nbr_fea_len": nbr_fea_len,
        **model_architecture,
    }

    train_ensemble(
        model_class=CrystalGraphConvNet,
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

    test_loader = DataLoader(
        test_set,
        **{**data_params, "batch_size": 64 * data_params["batch_size"], "shuffle": False},
    )

    results_dict = results_multitask(
        model_class=CrystalGraphConvNet,
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

    logits = results_dict["phdos_clf"]["logits"]
    targets = results_dict["phdos_clf"]["targets"]

    # calculate metrics and errors with associated errors for ensembles
    ens_logits = np.mean(logits, axis=0)

    ens_acc, *_, ens_roc_auc = get_metrics(targets, ens_logits, task).values()

    assert len(targets) == len(test_set) == len(test_idx)
    assert ens_acc > 0.85
    assert ens_roc_auc > 0.9


if __name__ == "__main__":
    pytest.main(["-v", __file__])
