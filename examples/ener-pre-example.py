import os
import sys
import argparse

import torch
from sklearn.model_selection import train_test_split as split

from roost.pretrain.ener_model import CrystalGraphPreNet
from roost.pretrain.ener_data import CrystalGraphData, collate_batch
from roost.utils import (
    train_ensemble,
    results_multitask,
)


def main(
    data_path,
    fea_path,
    targets,
    tasks,
    losses,
    robust,
    model_name="pre-cgcnn",
    elem_fea_len=64,
    n_graph=4,
    radius=5,
    max_num_nbr=12,
    dmin=0,
    step=0.2,
    p_mask=0.15,
    p_zero=0.8,
    ensemble=1,
    run_id=1,
    data_seed=42,
    epochs=100,
    patience=None,
    log=True,
    sample=1,
    val_size=0.2,
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

    assert len(targets) == len(tasks) == len(losses)

    assert (
        evaluate or train
    ), "No action given - At least one of 'train' or 'evaluate' cli flags required"

    if val_path:
        val_size = 0.0

    assert val_size < 1.0, (
        f"'val_size'({val_size}) must be less than 1"
    )

    if ensemble > 1 and (fine_tune or transfer):
        raise NotImplementedError(
            "If training an ensemble with fine tuning or transfering"
            " options the models must be trained one by one using the"
            " run-id flag."
        )

    assert not (
        fine_tune and transfer
    ), "Cannot fine-tune and transfer checkpoint(s) at the same time."

    task_dict = {k: v for k, v in zip(targets, tasks)}
    loss_dict = {k: v for k, v in zip(targets, losses)}

    dist_dict = {
        "radius": radius,
        "max_num_nbr": max_num_nbr,
        "dmin": dmin,
        "step": step,
        "p_mask": p_mask,
        "p_zero": p_zero,
    }

    dataset = CrystalGraphData(
        data_path=data_path, fea_path=fea_path, task_dict=task_dict, **dist_dict
    )

    n_targets = dataset.n_targets
    elem_emb_len = dataset.elem_fea_dim
    nbr_fea_len = dataset.nbr_fea_dim

    train_idx = list(range(len(dataset)))

    if train:
        if val_path:
            print(f"using independent validation set: {val_path}")
            val_set = CrystalGraphData(
                data_path=val_path, fea_path=fea_path, task_dict=task_dict, **dist_dict
            )
            val_set = torch.utils.data.Subset(val_set, range(len(val_set)))
        else:
            if val_size == 0.0:
                val_set = None
            else:
                print(f"using {val_size} of training set as validation set")
                train_idx, val_idx = split(
                    train_idx,
                    random_state=data_seed,
                    test_size=val_size,
                )
                val_set = torch.utils.data.Subset(dataset, val_idx)

        train_set = torch.utils.data.Subset(dataset, train_idx[0::sample])

    data_params = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": False,
        # "shuffle": False,
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

    if resume:
        resume = f"models/{model_name}/checkpoint-r{run_id}.pth.tar"

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
        "nbr_fea_len": nbr_fea_len,
        "elem_fea_len": elem_fea_len,
        "n_graph": n_graph,
    }

    os.makedirs(f"models/{model_name}/", exist_ok=True)

    if log:
        os.makedirs("runs/", exist_ok=True)

    os.makedirs("results/", exist_ok=True)

    if train:
        train_ensemble(
            model_class=CrystalGraphPreNet,
            model_name=model_name,
            run_id=run_id,
            ensemble_folds=ensemble,
            epochs=epochs,
            patience=patience,
            train_set=train_set,
            val_set=val_set,
            log=log,
            data_params=data_params,
            setup_params=setup_params,
            restart_params=restart_params,
            model_params=model_params,
            loss_dict=loss_dict,
        )


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(description=("cgcnn"))

    # data inputs
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/datasets/tests/cgcnn-regression.csv",
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
        default=0.2,
        type=float,
        metavar="FLOAT",
        help="Proportion of data used for validation",
    )

    # data embeddings
    parser.add_argument(
        "--fea-path",
        type=str,
        default="data/el-embeddings/cgcnn-embedding.json",
        # default="data/el-embeddings/megnet16-embedding.json",
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
        "--data-seed",
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

    # task inputs
    parser.add_argument(
        "--targets",
        nargs="*",
        type=str,
        metavar="STR",
        help="Task types for targets",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["regression"],
        type=str,
        metavar="STR",
        help="Task types for targets",
    )
    parser.add_argument(
        "--losses",
        nargs="*",
        default=["L1"],
        type=str,
        metavar="STR",
        help="Loss function if regression (default: 'L1')",
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
        default=4,
        type=int,
        metavar="INT",
        help="Number of message passing layers (default: 3)",
    )
    parser.add_argument(
        "--radius",
        default=5,
        type=float,
        metavar="FLOAT",
        help="Maximum radius for local neighbour graph (default: 5)",
    )
    parser.add_argument(
        "--max-num-nbr",
        default=12,
        type=int,
        metavar="INT",
        help="Maximum number of neighbours to consider (default: 12)",
    )
    parser.add_argument(
        "--dmin",
        default=0.0,
        type=float,
        metavar="FLOAT",
        help="Minimum distance of smeared gaussian basis (default 0.0)",
    )
    parser.add_argument(
        "--step",
        default=0.2,
        type=float,
        metavar="FLOAT",
        help="Step size of smeared gaussian basis (default: 0.2)",
    )
    parser.add_argument(
        "--p-mask",
        default=0.15,
        type=float,
        metavar="FLOAT",
        help="Proportion of crystal sites to mask (default: 0.15)",
    )
    parser.add_argument(
        "--p-zero",
        default=0.8,
        type=float,
        metavar="FLOAT",
        help="Proportion of masked sites to zero (default: 0.8)",
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
        default="pre-ener",
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
        args.model_name = f"{args.data_id}_s-{args.data_seed}_t-{args.sample}"

    assert all(
        [i in ["regression", "classification", "mask", "global"] for i in args.tasks]
    ), "Only `regression`, `classification`, `mask` and `global` are allowed as tasks"

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
