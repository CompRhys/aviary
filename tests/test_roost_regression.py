import numpy as np
import torch
from sklearn.model_selection import train_test_split as split

from aviary.roost.data import CompositionData, collate_batch
from aviary.roost.model import Roost
from aviary.utils import get_metrics, results_multitask, train_ensemble


def test_roost_regression(df_matbench_phonons):
    elem_emb = "matscholar200"
    targets = ["last phdos peak"]
    task = "regression"
    losses = ["L1"]
    robust = True
    model_name = "roost-reg-test"
    elem_fea_len = 64
    n_graph = 3
    ensemble = 2
    run_id = 1
    data_seed = 42
    epochs = 25
    log = False
    sample = 1
    test_size = 0.2
    resume = False
    fine_tune = None
    transfer = None
    optim = "AdamW"
    learning_rate = 3e-4
    momentum = 0.9
    weight_decay = 1e-6
    batch_size = 128
    workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task_dict = dict(zip(targets, [task]))
    loss_dict = dict(zip(targets, losses))

    dataset = CompositionData(
        df=df_matbench_phonons, elem_emb=elem_emb, task_dict=task_dict
    )
    n_targets = dataset.n_targets
    elem_emb_len = dataset.elem_emb_len

    train_idx = list(range(len(dataset)))

    print(f"using {test_size} of training set as test set")
    train_idx, test_idx = split(train_idx, random_state=data_seed, test_size=test_size)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    print("No validation set used, using test set for evaluation purposes")
    # NOTE that when using this option care must be taken not to
    # peak at the test-set. The only valid model to use is the one
    # obtained after the final epoch where the epoch count is
    # decided in advance of the experiment.
    val_set = test_set

    train_set = torch.utils.data.Subset(dataset, train_idx[0::sample])

    data_params = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": collate_batch,
    }

    setup_params = {
        "optim": optim,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "device": device,
    }

    restart_params = {
        "resume": resume,
        "fine_tune": fine_tune,
        "transfer": transfer,
    }

    model_params = {
        "task_dict": task_dict,
        "robust": robust,
        "n_targets": n_targets,
        "elem_emb_len": elem_emb_len,
        "elem_fea_len": elem_fea_len,
        "n_graph": n_graph,
        "elem_heads": 2,
        "elem_gate": [256],
        "elem_msg": [256],
        "cry_heads": 2,
        "cry_gate": [256],
        "cry_msg": [256],
        "trunk_hidden": [256, 256],
        "out_hidden": [128, 64],
    }

    train_ensemble(
        model_class=Roost,
        model_name=model_name,
        run_id=run_id,
        ensemble_folds=ensemble,
        epochs=epochs,
        train_set=train_set,
        val_set=val_set,
        log=log,
        data_params=data_params,
        setup_params=setup_params,
        restart_params=restart_params,
        model_params=model_params,
        loss_dict=loss_dict,
    )

    data_params["batch_size"] = 64 * batch_size  # faster model inference
    data_params["shuffle"] = False  # need fixed data order due to ensembling

    results_dict = results_multitask(
        model_class=Roost,
        model_name=model_name,
        run_id=run_id,
        ensemble_folds=ensemble,
        test_set=test_set,
        data_params=data_params,
        robust=robust,
        task_dict=task_dict,
        device=device,
        eval_type="checkpoint",
        save_results=False,
    )

    pred = results_dict["last phdos peak"]["pred"]
    target = results_dict["last phdos peak"]["target"]

    y_ens = np.mean(pred, axis=0)

    mae, rmse, r2 = get_metrics(target, y_ens, task).values()

    assert r2 > 0.7
    assert mae < 150
    assert rmse < 300
