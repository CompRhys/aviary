import os

import torch
from sklearn.model_selection import train_test_split as split

from roost.roost.data import CompositionData, collate_batch
from roost.roost.model import Roost
from roost.utils import results_regression, train_ensemble

torch.manual_seed(0)  # ensure reproducible results


def test_single_roost():

    data_path = "data/datasets/expt-non-metals.csv"
    fea_path = "data/embeddings/matscholar-embedding.json"
    task = "regression"
    loss = "L1"
    robust = True
    model_name = "roost"
    elem_fea_len = 64
    n_graph = 3
    ensemble = 1
    run_id = 1
    data_seed = 42
    epochs = 10
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = CompositionData(data_path=data_path, fea_path=fea_path, task=task)
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
        "loss": loss,
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
        "task": task,
        "robust": robust,
        "n_targets": n_targets,
        "elem_emb_len": elem_emb_len,
        "elem_fea_len": elem_fea_len,
        "n_graph": n_graph,
        "elem_heads": 3,
        "elem_gate": [256],
        "elem_msg": [256],
        "cry_heads": 3,
        "cry_gate": [256],
        "cry_msg": [256],
        "out_hidden": [1024, 512, 256, 128, 64],
    }

    os.makedirs(f"models/{model_name}", exist_ok=True)
    os.makedirs(f"results/{model_name}", exist_ok=True)

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
    )

    data_params["batch_size"] = 64 * batch_size  # faster model inference
    data_params["shuffle"] = False  # need fixed data order due to ensembling

    r2, mae, rmse = results_regression(
        model_class=Roost,
        model_name=model_name,
        run_id=run_id,
        ensemble_folds=ensemble,
        test_set=test_set,
        data_params=data_params,
        robust=robust,
        device=device,
        eval_type="checkpoint",
    )

    assert r2 > 0.7
    assert mae < 0.55
    assert rmse < 0.83
    # standard values after 10 epochs
    # - R2 Score: 0.7017
    # - MAE: 0.5470
    # - RMSE: 0.8297
