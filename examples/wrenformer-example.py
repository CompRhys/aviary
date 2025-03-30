import argparse

import pandas as pd
import torch
from sklearn.model_selection import train_test_split as split

from aviary.utils import results_multitask, train_ensemble
from aviary.wrenformer.data import df_to_in_mem_dataloader
from aviary.wrenformer.model import Wrenformer


def main(
    data_path,
    targets,
    tasks,
    losses,
    robust,
    model_name="wrenformer",
    embedding_type="protostructure",
    n_attn_layers=6,
    n_attn_heads=4,
    d_model=128,
    ensemble=1,
    run_id=1,
    data_seed=42,
    epochs=100,
    log=False,
    sample=1,
    test_size=0.2,
    test_path=None,
    val_size=0,
    val_path=None,
    resume=False,
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
    device=None,
    **kwargs,
):
    """Train and evaluate a Wrenformer model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"The model will run on the {device} device")

    if not len(targets) == len(tasks) == len(losses):
        raise AssertionError

    if not (evaluate or train):
        raise AssertionError(
            "No action given - At least one of 'train' or 'evaluate' cli flags required"
        )

    if test_size + val_size >= 1:
        raise AssertionError(
            f"'test_size'({test_size}) plus 'val_size'({val_size}) must be less than 1"
        )

    task_dict = dict(zip(targets, tasks, strict=False))
    loss_dict = dict(zip(targets, losses, strict=False))

    # Load and preprocess data
    df = pd.read_csv(data_path, keep_default_na=False, na_values=[])

    # Split datasets
    if evaluate:
        if test_path:
            print(f"using independent test set: {test_path}")
            test_df = pd.read_csv(test_path, keep_default_na=False, na_values=[])
        else:
            print(f"using {test_size} of training set as test set")
            train_df, test_df = split(df, random_state=data_seed, test_size=test_size)

    if train:
        if val_path:
            print(f"using independent validation set: {val_path}")
            val_df = pd.read_csv(val_path, keep_default_na=False, na_values=[])
        elif val_size == 0 and evaluate:
            print("No validation set used, using test set for evaluation purposes")
            val_df = test_df
        elif val_size == 0:
            val_df = None
        else:
            print(f"using {val_size} of training set as validation set")
            test_size = val_size / (1 - test_size)
            train_df, val_df = split(
                train_df, random_state=data_seed, test_size=test_size
            )

    # Setup data loaders
    data_loader_kwargs = dict(
        id_col="material_id",
        input_col="protostructure",
        target_col=targets[0],
        embedding_type=embedding_type,
        device=device,
    )

    if train:
        if sample > 1:
            train_df = train_df.iloc[::sample].copy()

        train_loader = df_to_in_mem_dataloader(
            train_df,
            batch_size=batch_size,
            shuffle=True,
            **data_loader_kwargs,
        )

        val_loader = df_to_in_mem_dataloader(
            val_df,
            batch_size=batch_size * 16,
            shuffle=False,
            **data_loader_kwargs,
        )

        # Model parameters
        n_targets = [
            1 if task_type == "regression" else train_df[target_col].max() + 1
            for target_col, task_type in task_dict.items()
        ]

        model_params = {
            "task_dict": task_dict,
            "robust": robust,
            "n_targets": n_targets,
            "n_features": train_loader.tensors[0][0].shape[-1],
            "d_model": d_model,
            "n_attn_layers": n_attn_layers,
            "n_attn_heads": n_attn_heads,
            "trunk_hidden": (1024, 512),
            "out_hidden": (256, 128, 64),
            "embedding_aggregations": ("mean",),
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

        train_ensemble(
            model_class=Wrenformer,
            model_name=model_name,
            run_id=run_id,
            ensemble_folds=ensemble,
            epochs=epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            log=log,
            setup_params=setup_params,
            restart_params=restart_params,
            model_params=model_params,
            loss_dict=loss_dict,
        )

    if evaluate:
        test_loader = df_to_in_mem_dataloader(
            test_df,
            batch_size=batch_size * 64,
            shuffle=False,
            **data_loader_kwargs,
        )

        results_multitask(
            model_class=Wrenformer,
            model_name=model_name,
            run_id=run_id,
            ensemble_folds=ensemble,
            test_loader=test_loader,
            robust=robust,
            task_dict=task_dict,
            device=device,
            eval_type="checkpoint",
            save_results=False,
        )


def input_parser():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description=("Wrenformer"))

    # data inputs
    parser.add_argument(
        "--data-path",
        metavar="PATH",
        help="Path to main data set/training set",
    )
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument(
        "--val-path", metavar="PATH", help="Path to independent validation set"
    )
    valid_group.add_argument(
        "--val-size",
        default=0,
        type=float,
        metavar="FLOAT",
        help="Proportion of data used for validation",
    )
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--test-path", metavar="PATH", help="Path to independent test set"
    )
    test_group.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        metavar="FLOAT",
        help="Proportion of data set for testing",
    )

    # data loader inputs
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
        "--targets", nargs="+", metavar="STR", help="Task types for targets"
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=("regression", "classification"),
        default=["regression"],
        metavar="STR",
        help="Task types for targets",
    )
    parser.add_argument(
        "--losses",
        nargs="*",
        choices=("L1", "L2", "CSE"),
        default=["L1"],
        metavar="STR",
        help="Loss function if regression (default: 'L1')",
    )

    # optimizer inputs
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
        help="Specifies whether to use heteroscedastic loss variants",
    )
    parser.add_argument(
        "--optim",
        default="AdamW",
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
        default=None,
        metavar="STR",
        help="Name for sub-directory where models will be stored",
    )
    name_group.add_argument(
        "--data-id",
        default="wren",
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
        "--fine-tune", metavar="PATH", help="Checkpoint path for fine tuning"
    )
    use_group.add_argument(
        "--transfer", metavar="PATH", help="Checkpoint path for transfer learning"
    )
    use_group.add_argument(
        "--resume", action="store_true", help="Resume from previous checkpoint"
    )

    # task type
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate the model/ensemble"
    )
    parser.add_argument("--train", action="store_true", help="Train the model/ensemble")

    # misc
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--log", action="store_true", help="Log training metrics to TensorBoard"
    )

    # model architecture inputs
    parser.add_argument(
        "--embedding-type",
        default="protostructure",
        type=str,
        metavar="STR",
        help="Type of embedding to use (default: 'protostructure')",
    )
    parser.add_argument(
        "--n-attn-layers",
        default=6,
        type=int,
        metavar="INT",
        help="Number of attention layers (default: 6)",
    )
    parser.add_argument(
        "--n-attn-heads",
        default=4,
        type=int,
        metavar="INT",
        help="Number of attention heads per layer (default: 4)",
    )
    parser.add_argument(
        "--d-model",
        default=128,
        type=int,
        metavar="INT",
        help="Dimension of model embeddings (default: 128)",
    )

    args = parser.parse_args()

    if args.model_name is None:
        args.model_name = f"{args.data_id}_s-{args.data_seed}_t-{args.sample}"

    args.device = (
        torch.device("cuda")
        if (not args.disable_cuda) and torch.cuda.is_available()
        else torch.device("cpu")
    )

    return args


if __name__ == "__main__":
    args = input_parser()
    raise SystemExit(main(**vars(args)))
