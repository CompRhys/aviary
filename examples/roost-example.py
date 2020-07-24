import os
import sys
import argparse

import torch
from sklearn.model_selection import train_test_split as split

from roost.roost.model import Roost
from roost.roost.data import CompositionData, collate_batch
from roost.utils import (
    train_ensemble,
    results_regression,
    results_classification,
)


def main(
    data_path,
    fea_path,
    task,
    loss,
    robust,
    model_name="roost",
    elem_fea_len=64,
    n_graph=3,
    ensemble=1,
    run_id=1,
    seed=42,
    epochs=100,
    log=True,
    sample=1,
    test_size=0.2,
    test_path=None,
    val_size=0.0,
    val_path=None,
    resume=None,
    fine_tune=None,
    transfer=None,
    train=True,
    evaluate=True,
    optim="AdamW",
    learning_rate=3e-4,
    momentum=0.9,
    weight_decay=1e-6,
    batch_size=128,
    workers=0,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    **kwargs,
):
    assert evaluate or train, (
        "No task given - Set at least one of 'train' or 'evaluate' kwargs as True"
    )
    assert task in ["regression", "classification"], (
        "Only 'regression' or 'classification' allowed for 'task'"
    )

    if test_path:
        test_size = 0.0

    if not (test_path and val_path):
        assert test_size + val_size < 1.0, (
            f"'test_size'({test_size}) "
            f"plus 'val_size'({val_size}) must be less than 1"
        )

    if ensemble > 1 and (fine_tune or transfer):
        raise NotImplementedError(
            "If training an ensemble with fine tuning or transfering"
            " options the models must be trained one by one using the"
            " run-id flag."
        )

    assert not (fine_tune and transfer), (
        "Cannot fine-tune and" " transfer checkpoint(s) at the same time."
    )

    dataset = CompositionData(data_path=data_path, fea_path=fea_path, task=task)
    n_targets = dataset.n_targets
    elem_emb_len = dataset.elem_emb_len

    train_idx = list(range(len(dataset)))

    if evaluate:
        if test_path:
            print(f"using independent test set: {test_path}")
            test_set = CompositionData(
                data_path=test_path, fea_path=fea_path, task=task
            )
            test_set = torch.utils.data.Subset(test_set, range(len(test_set)))
        elif test_size == 0.0:
            raise ValueError("test-size must be non-zero to evaluate model")
        else:
            print(f"using {test_size} of training set as test set")
            train_idx, test_idx = split(
                train_idx, random_state=seed, test_size=test_size
            )
            test_set = torch.utils.data.Subset(dataset, test_idx)

    if train:
        if val_path:
            print(f"using independent validation set: {val_path}")
            val_set = CompositionData(data_path=val_path, fea_path=fea_path, task=task)
            val_set = torch.utils.data.Subset(val_set, range(len(val_set)))
        else:
            if val_size == 0.0 and evaluate:
                print("No validation set used, using test set for evaluation purposes")
                # NOTE that when using this option care must be taken not to
                # peak at the test-set. The only valid model to use is the one
                # obtained after the final epoch where the epoch count is
                # decided in advance of the experiment.
                val_set = test_set
            elif val_size == 0.0:
                val_set = None
            else:
                print(f"using {val_size} of training set as validation set")
                train_idx, val_idx = split(
                    train_idx, random_state=seed, test_size=val_size / (1 - test_size),
                )
                val_set = torch.utils.data.Subset(dataset, val_idx)

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

    os.makedirs(f"models/{model_name}/", exist_ok=True)

    if log:
        os.makedirs("runs/", exist_ok=True)

    os.makedirs("results/", exist_ok=True)

    if train:
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

    if evaluate:

        data_reset = {
            "batch_size": 16 * batch_size,  # faster model inference
            "shuffle": False,  # need fixed data order due to ensembling
        }
        data_params.update(data_reset)

        if task == "regression":
            results_regression(
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
        elif task == "classification":
            results_classification(
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


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(
        description=(
            "Roost - a Structure Agnostic Message Passing "
            "Neural Network for Inorganic Materials"
        )
    )

    # data inputs
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/datasets/expt-non-metals.csv",
        metavar="PATH",
        help="Path to main data set/training set",
    )
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument(
        "--val-path",
        type=str,
        metavar="PATH",
        help="Path to independent validation set",
    )
    valid_group.add_argument(
        "--val-size",
        default=0.0,
        type=float,
        metavar="FLOAT",
        help="Proportion of data used for validation",
    )
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--test-path",
        type=str,
        metavar="PATH",
        help="Path to independent test set"
    )
    test_group.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        metavar="FLOAT",
        help="Proportion of data set for testing",
    )

    # data embeddings
    parser.add_argument(
        "--fea-path",
        type=str,
        default="data/embeddings/matscholar-embedding.json",
        metavar="PATH",
        help="Element embedding feature path",
    )

    # dataloader inputs
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        metavar="INT",
        help="Number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        "--bsize",
        default=128,
        type=int,
        metavar="INT",
        help="Mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        metavar="INT",
        help="Seed used when splitting data sets (default: 0)",
    )
    parser.add_argument(
        "--sample",
        default=1,
        type=int,
        metavar="INT",
        help="Sub-sample the training set for learning curves",
    )

    # optimiser inputs
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="INT",
        help="Number of training epochs to run (default: 100)",
    )
    parser.add_argument(
        "--loss",
        default="L1",
        type=str,
        metavar="STR",
        help="Loss function if regression (default: 'L1')",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Specifies whether to use hetroskedastic loss variants",
    )
    parser.add_argument(
        "--optim",
        default="AdamW",
        type=str,
        metavar="STR",
        help="Optimizer used for training (default: 'AdamW')",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        default=3e-4,
        type=float,
        metavar="FLOAT",
        help="Initial learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="FLOAT [0,1]",
        help="Optimizer momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-6,
        type=float,
        metavar="FLOAT [0,1]",
        help="Optimizer weight decay (default: 1e-6)",
    )

    # graph inputs
    parser.add_argument(
        "--elem-fea-len",
        default=64,
        type=int,
        metavar="INT",
        help="Number of hidden features for elements (default: 64)",
    )
    parser.add_argument(
        "--n-graph",
        default=3,
        type=int,
        metavar="INT",
        help="Number of message passing layers (default: 3)",
    )

    # ensemble inputs
    parser.add_argument(
        "--ensemble",
        default=1,
        type=int,
        metavar="INT",
        help="Number models to ensemble",
    )
    name_group = parser.add_mutually_exclusive_group()
    name_group.add_argument(
        "--model-name",
        type=str,
        default=None,
        metavar="STR",
        help="Name for sub-directory where models will be stored",
    )
    name_group.add_argument(
        "--data-id",
        default="roost",
        type=str,
        metavar="STR",
        help="Partial identifier for sub-directory where models will be stored",
    )
    parser.add_argument(
        "--run-id",
        default=0,
        type=int,
        metavar="INT",
        help="Index for model in an ensemble of models",
    )

    # restart inputs
    use_group = parser.add_mutually_exclusive_group()
    use_group.add_argument(
        "--fine-tune",
        type=str,
        metavar="PATH",
        help="Checkpoint path for fine tuning"
    )
    use_group.add_argument(
        "--transfer",
        type=str,
        metavar="PATH",
        help="Checkpoint path for transfer learning",
    )
    use_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint"
    )

    # task type
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--classification",
        action="store_true",
        help="Specifies a classification task"
    )
    task_group.add_argument(
        "--regression",
        action="store_true",
        help="Specifies a regression task"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model/ensemble",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model/ensemble"
    )

    # misc
    parser.add_argument(
        "--disable-cuda",
        action="store_true",
        help="Disable CUDA"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log training metrics to tensorboard"
    )

    args = parser.parse_args(sys.argv[1:])

    if args.model_name is None:
        args.model_name = f"{args.data_id}_s-{args.seed}_t-{args.sample}"

    if args.regression:
        args.task = "regression"
    elif args.classification:
        args.task = "classification"
    else:
        args.task = "regression"

    args.device = (
        torch.device("cuda")
        if (not args.disable_cuda) and torch.cuda.is_available()
        else torch.device("cpu")
    )

    return args


if __name__ == "__main__":
    args = input_parser()

    print(f"The model will run on the {args.device} device")

    main(**vars(args))
