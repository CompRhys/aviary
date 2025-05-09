import os
from copy import deepcopy
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from aviary import ROOT
from aviary.core import BaseModelClass, Normalizer, TaskType, np_softmax
from aviary.data import InMemoryDataLoader
from aviary.losses import robust_l1_loss
from aviary.predict import make_ensemble_predictions
from aviary.utils import get_metrics, print_walltime
from aviary.wrenformer.data import df_to_in_mem_dataloader
from aviary.wrenformer.model import Wrenformer

torch.manual_seed(0)  # ensure reproducible results

reg_key, clf_key = "regression", "classification"


def lr_lambda(epoch: int) -> float:
    """This lambda goes up linearly until warmup_steps, then follows a power law decay.
    Acts as a prefactor to the learning rate, i.e. actual_lr = lr_lambda(epoch) *
    learning_rate.
    """
    warmup_steps = 10
    return min((epoch + 1) ** (-0.5), (epoch + 1) * warmup_steps ** (-1.5))


@print_walltime(end_desc="train_model()")
def train_model(
    run_name: str,
    model: BaseModelClass,
    epochs: int,
    target_col: str,
    task_type: TaskType,
    train_loader: DataLoader | InMemoryDataLoader,
    test_loader: DataLoader | InMemoryDataLoader,
    *,  # force keyword-only arguments
    checkpoint: Literal["local", "wandb"] | None = None,
    checkpoint_frequency: int = 10,
    learning_rate: float = 1e-4,
    model_params: dict[str, Any] | None = None,
    run_params: dict[str, Any] | None = None,
    optimizer: str | tuple[str, dict] = "AdamW",
    scheduler: str | tuple[str, dict] = "LambdaLR",
    swa_start: float | None = None,
    swa_lr: float | None = None,
    test_df: pd.DataFrame = None,
    timestamp: str | None = None,
    verbose: bool = False,
    wandb_path: str | None = None,
    wandb_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any], pd.DataFrame]:
    """Core training function. Handles checkpointing and metric logging.
    Wrapped by other functions like train_wrenformer() for specific datasets.

    Args:
        run_name (str): A string to describe the training run. Should usually contain
            model type (Roost/Wren) and important params. Include 'robust' to use a
            robust loss function and have the model learn to predict an aleatoric
            uncertainty.
        model (BaseModelClass): A model instance subclassing aviary.core.BaseModelClass.
        epochs (int): How many epochs to train for. Defaults to 100.
        target_col (str): Name of df column containing the target values.
        task_type ('regression' | 'classification'): What type of task to train the
            model for.
        train_loader (DataLoader | InMemoryDataLoader): Training data loader.
        test_loader (DataLoader | InMemoryDataLoader): Test data loader.
        checkpoint (None | 'local' | 'wandb'): Whether to save the model, optimizer,
            and scheduler state dicts to disk (local) or upload to WandB.
            Defaults to None. To later copy a wandb checkpoint file to cwd and use it:
            ```py
            run_path = "<user|team>/<project>/<run_id>"  # e.g. aviary/matbench/31qh7b5q
            checkpoint = wandb.restore("checkpoint.pth", run_path)
            torch.load(checkpoint.name)
            ```
        checkpoint_frequency (int): How often to save a checkpoint. Defaults to 10.
        learning_rate (float): The optimizer's learning rate. Defaults to 1e-4.
        model_params (dict): Arguments passed to model class. E.g. dict(n_attn_layers=6,
            embedding_aggregation=("mean", "std")) for Wrenformer.
        run_params (dict[str, Any]): Additional parameters to merge into the run's dict of
            model_params. Will be logged to wandb. Can be anything really. Defaults to {}.
        optimizer (str | tuple[str, dict]): Name of a torch.optim.Optimizer class like
            'Adam', 'AdamW', 'SGD', etc. Can be a string or a string and dict with params
            to pass to the class. Defaults to 'AdamW'.
        scheduler (str | tuple[str, dict]): Name of a torch.optim.lr_scheduler class like
            'LambdaLR', 'StepLR', 'CosineAnnealingLR', etc. Can be a string to create a
            scheduler with default values or tuple[str, dict] with custom params.
            E.g. ('CosineAnnealingLR', {'T_max': n_epochs}). Defaults to 'LambdaLR'.
            See https://stackoverflow.com/a/2121918 about pickle errors when trying to
            load a LambdaLR scheduler from a torch.save() checkpoint created prior to this
            file having been renamed.
        swa_start (float | None): When to start using stochastic weight averaging during
            training. Should be a float between 0 and 1. 0.7 means start SWA after 70%
            of epochs. Set to None to disable SWA. Defaults to None. Proposed in
            https://arxiv.org/abs/1803.05407.
        swa_lr (float | None): Learning rate for SWA scheduler. Defaults to learning_rate.
        test_df (pd.DataFrame): Test data as a DataFrame. Model preds will be inserted
            as new columns and df returned.
        timestamp (str | None): Will prefix the names of model checkpoint files and other
            output files. Will also be included in run_params. Defaults to None.
        verbose (bool): Whether to print progress and metrics to stdout. Defaults to
            False.
        wandb_path (str | None): Path to Weights and Biases project where to log this run
            formatted as '<entity>/<project>'. Defaults to None which means logging is
            disabled.
        wandb_kwargs (dict[str, Any] | None): Kwargs to pass to wandb.init() like
            dict(tags=['ensemble-id-1']). Should not include keys config, project, entity
            as they're already set by this function. Defaults to None.

    Raises:
        ValueError: On unknown dataset_name or invalid checkpoint.

    Returns:
        tuple[dict[str, float], dict[str, Any], pd.DataFrame]: A tuple containing:
            - Test set metrics dictionary
            - Run hyperparameters dictionary
            - Test dataframe with predictions
    """
    if checkpoint not in (None, "local", "wandb"):
        raise ValueError(f"Unknown {checkpoint=}")
    if checkpoint == "wandb" and not wandb_path:
        raise ValueError(f"Cannot save checkpoint to wandb if {wandb_path=}")

    robust = "robust" in run_name.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pytorch running on {device=}")

    loss_func = (
        (robust_l1_loss if robust else torch.nn.L1Loss())
        if task_type == reg_key
        else (torch.nn.NLLLoss() if robust else torch.nn.CrossEntropyLoss())
    )
    loss_dict = {target_col: (task_type, loss_func)}

    normalizer_dict = {target_col: Normalizer() if task_type == reg_key else None}
    # TODO consider actually fitting the normalizer, currently just passed into
    # model.evaluate() to match function signature

    # embedding_len is the length of the embedding vector for a Wyckoff position
    # encoding the element type (usually 200-dim Matscholar embeddings) and Wyckoff
    # position (see 'bra-alg-off.json') + 1 for the weight of that Wyckoff position (or
    # element) in the material embedding_len = train_loader.tensors[0][0].shape[-1]
    # # Roost and Wren embedding size resp.
    # assert embedding_len in (200 + 1, 200 + 1 + 444), f"{embedding_len=}"

    model.to(device)
    if isinstance(optimizer, str):
        optimizer_name, optimizer_params = optimizer, None
    elif isinstance(optimizer, tuple | list):
        optimizer_name, optimizer_params = optimizer
    else:
        raise TypeError(f"Unknown {optimizer=}")

    optimizer_cls = getattr(torch.optim, optimizer_name)
    optimizer_instance = optimizer_cls(
        params=model.parameters(), lr=learning_rate, **(optimizer_params or {})
    )

    if scheduler == "LambdaLR":
        scheduler_name, scheduler_params = "LambdaLR", {"lr_lambda": lr_lambda}
    elif isinstance(scheduler, str):
        scheduler_name, scheduler_params = scheduler, None
    elif isinstance(scheduler, tuple | list):
        scheduler_name, scheduler_params = scheduler
    else:
        raise ValueError(f"Unknown {scheduler=}")

    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
    lr_scheduler = scheduler_cls(optimizer_instance, **(scheduler_params or {}))

    if swa_start is not None:
        swa_lr = swa_lr or learning_rate
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer_instance, swa_lr=swa_lr)

    run_params = dict(
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer=dict(name=optimizer_name, params=optimizer_params),
        lr_scheduler=dict(name=scheduler_name, params=scheduler_params),
        target=target_col,
        robust=robust,
        # embedding_len=embedding_len,
        losses=loss_dict,
        trainable_params=model.num_params,
        task_type=task_type,
        checkpoint=checkpoint,
        **(run_params or {}),
    )
    if swa_start:
        run_params["swa"] = dict(
            start=swa_start, epochs=int(swa_start * epochs), learning_rate=swa_lr
        )
    if task_type == reg_key and hasattr(train_loader, "df"):
        train_df = getattr(train_loader, "df", train_loader.dataset.df)  # type: ignore[union-attr]
        targets = train_df[target_col]
        run_params["dummy_mae"] = (targets - targets.mean()).abs().mean()
    if timestamp:
        run_params["timestamp"] = timestamp
    for x in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID"):
        if x in os.environ:
            run_params[x.lower()] = os.environ[x]
    print(f"{run_params=}")

    if wandb_path:
        if wandb.run is None:
            wandb.login()
        wandb_entity, wandb_project = wandb_path.split("/")
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            # https://docs.wandb.ai/guides/track/launch#init-start-error
            settings=wandb.Settings(start_method="fork"),
            name=run_name,
            config=run_params,
            **wandb_kwargs or {},
        )

    for epoch in tqdm(range(1, epochs + 1), disable=None, desc="Training epoch"):
        train_metrics = model.evaluate(
            train_loader,
            loss_dict,
            optimizer_instance,
            normalizer_dict,
            action="train",
            verbose=verbose,
        )

        with torch.no_grad():
            val_metrics = model.evaluate(
                test_loader,
                loss_dict,
                None,
                normalizer_dict,
                action="evaluate",
                verbose=verbose,
            )

        if swa_start and epoch >= int(swa_start * epochs):
            if epoch == int(swa_start * epochs):
                print("Starting stochastic weight averaging...")
            swa_model.update_parameters(model)  # type: ignore[reportPossiblyUnboundVariable]
            swa_scheduler.step()  # type: ignore[reportPossiblyUnboundVariable]
        elif scheduler_name == "ReduceLROnPlateau":
            val_metric = val_metrics[target_col][
                "MAE" if task_type == reg_key else "Accuracy"
            ]
            lr_scheduler.step(val_metric)
        else:
            lr_scheduler.step()

        model.epoch += 1

        if wandb_path:
            wandb.log({"training": train_metrics, "validation": val_metrics})

        if epoch % checkpoint_frequency == 0 and epoch < epochs:
            inference_model = swa_model if swa_start else model  # type: ignore[reportPossiblyUnboundVariable]
            inference_model.eval()
            checkpoint_model(
                checkpoint_endpoint=checkpoint,
                model_params=model_params,
                inference_model=inference_model,
                optimizer_instance=optimizer_instance,
                lr_scheduler=lr_scheduler,
                loss_dict=loss_dict,
                epochs=epoch,
                test_metrics=val_metrics,
                timestamp=timestamp,
                run_name=run_name,
                normalizer_dict=normalizer_dict,
                run_params=run_params,
                scheduler_name=scheduler_name,
            )

    # get test set predictions
    if swa_start is not None:
        n_swa_epochs = int((1 - swa_start) * epochs)
        print(
            f"Using SWA model with weights averaged over {n_swa_epochs} epochs "
            f"({swa_start=})"
        )

    inference_model = swa_model if swa_start else model  # type: ignore[reportPossiblyUnboundVariable]
    inference_model.eval()

    with torch.no_grad():
        preds = np.concatenate(
            [
                inference_model(
                    *[
                        tensor.to(inference_model.device)
                        if hasattr(tensor, "to")
                        else tensor
                        for tensor in inputs
                    ]
                )[0]
                .cpu()
                .numpy()
                for inputs, *_ in test_loader
            ]
        ).squeeze()

    if test_df is None:
        if not isinstance(test_loader, DataLoader):
            raise TypeError(f"Unknown {test_loader=}")
        test_df = test_loader.dataset.df

    if robust:
        preds, aleatoric_log_std = np.split(preds, 2, axis=1)
        preds = preds.squeeze()
        aleatoric_std = np.exp(aleatoric_log_std.squeeze())
        df_std = pd.DataFrame(aleatoric_std, index=test_df.index).add_prefix(
            "aleatoric_std_"
        )
        test_df[df_std.columns] = df_std
    if task_type == clf_key:
        preds = np_softmax(preds, axis=1)

    targets = test_df[target_col]
    # preds can have shape (n_samples, n_classes) if doing multi-class classification so
    # use df to merge all columns into test_df
    df_preds = pd.DataFrame(preds, index=test_df.index).add_prefix(f"{target_col}_pred_")
    test_df[df_preds.columns] = df_preds  # requires shuffle=False for test_loader

    test_metrics = get_metrics(targets, preds, task_type)

    # save model checkpoint
    if checkpoint is not None:
        checkpoint_model(
            checkpoint_endpoint=checkpoint,
            model_params=model_params,
            inference_model=inference_model,
            optimizer_instance=optimizer_instance,
            lr_scheduler=lr_scheduler,
            loss_dict=loss_dict,
            epochs=epochs,
            test_metrics=test_metrics,
            timestamp=timestamp,
            run_name=run_name,
            normalizer_dict=normalizer_dict,
            run_params=run_params,
            scheduler_name=scheduler_name,
        )

    # record test set metrics and scatter/ROC plots to wandb
    if wandb_path:
        wandb.run.summary["test"] = test_metrics  # type: ignore[union-attr]
        wandb_table = wandb.Table(dataframe=test_df.filter(regex="^((?!structure).)"))
        if task_type == reg_key:
            from sklearn.metrics import r2_score

            MAE = np.abs(targets - preds).mean()
            R2 = r2_score(targets, preds)
            scatter_plot = wandb.plot_table(
                vega_spec_name="janosh/scatter-parity",
                data_table=wandb_table,
                fields=dict(x=target_col, y=test_df.filter(like="_pred_").columns[0]),
                string_fields=dict(title=f"{run_name}\n{MAE=:.4}\n{R2=:.4}"),
            )
            wandb.log({"true_pred_scatter": scatter_plot})
        elif task_type == clf_key:
            from sklearn.metrics import accuracy_score, roc_auc_score

            ROCAUC = roc_auc_score(targets, preds[:, 1])
            accuracy = accuracy_score(targets, preds.argmax(axis=1))
            title = f"{run_name}\n{accuracy=:.4}\n{ROCAUC=:.4}"
            roc_curve = wandb.plot.roc_curve(targets, preds, title=title)
            wandb.log({"roc_curve": roc_curve})

        wandb.finish()

    return test_metrics, run_params, test_df


def checkpoint_model(
    checkpoint_endpoint: Literal["local", "wandb"] | None,
    model_params: dict | None,
    inference_model: nn.Module,
    optimizer_instance: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    loss_dict: dict,
    epochs: int,
    test_metrics: dict,
    timestamp: str | None,
    run_name: str,
    normalizer_dict: dict,
    run_params: dict,
    scheduler_name: str,
) -> None:
    """Save model checkpoint to different endpoints."""
    if checkpoint_endpoint is None:
        return

    if model_params is None:
        raise ValueError("Must provide model_params to save checkpoint, got None")

    checkpoint_dict = dict(
        model_params=model_params,
        model_state=inference_model.state_dict(),
        optimizer_state=optimizer_instance.state_dict(),
        scheduler_state=lr_scheduler.state_dict(),
        loss_dict=loss_dict,
        epoch=epochs,
        metrics=test_metrics,
        run_name=run_name,
        normalizer_dict=normalizer_dict,
        run_params=deepcopy(run_params),
    )
    if scheduler_name == "LambdaLR":
        # exclude lr_lambda from pickled checkpoint since it causes errors when
        # torch.load()-ing a checkpoint and the file defining lr_lambda() was
        # renamed
        checkpoint_dict["run_params"]["lr_scheduler"].pop("params")

    if checkpoint_endpoint == "local":
        os.makedirs(f"{ROOT}/models", exist_ok=True)
        checkpoint_path = (
            f"{ROOT}/models/{timestamp + '-' if timestamp else ''}{run_name}-{epochs}.pth"
        )
        torch.save(checkpoint_dict, checkpoint_path)

    if checkpoint_endpoint == "wandb":
        if wandb.run is None:
            raise ValueError(
                "can't save model checkpoint to Weights and Biases, wandb.run is None"
            )
        torch.save(
            checkpoint_dict,
            f"{wandb.run.dir}/{timestamp + '-' if timestamp else ''}{run_name}-{epochs}.pth",  # noqa: E501
        )


def train_wrenformer(
    run_name: str,
    target_col: str,
    task_type: TaskType,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 128,
    inference_multiplier: int = 4,
    embedding_type: str | None = None,
    id_col: str = "material_id",
    input_col: str | None = None,
    model_params: dict[str, Any] | None = None,
    data_loader_device: str = "cpu",
    **kwargs: Any,
) -> tuple[dict[str, float], dict[str, Any], pd.DataFrame]:
    """Train a Wrenformer model on a dataframe. This function handles the DataLoader
    creation, then delegates to train_model().

    Args:
        run_name (str): A string to describe the training run. Should usually contain
            model type (Roost/Wren) and important params. Include 'robust' to use a
            robust loss function and have the model learn to predict an aleatoric
            uncertainty.
        target_col (str): Column name in train_df and test_df containing target values.
        task_type ('regression' | 'classification'): What type of task to train the
            model for.
        train_df (pd.DataFrame): Training set dataframe.
        test_df (pd.DataFrame): Test set dataframe.
        batch_size (int, optional): Batch size for training. Defaults to 128.
        inference_multiplier (int, optional): Multiplier for the test set data loader
            batch size. Defaults to 1.
        embedding_type ('protostructure' | 'composition', optional): Type of
            embedding to use. Defaults to None meaning auto-detect based on 'wren'/'roost'
            in run_name.
        id_col (str, optional): Column name in train_df and test_df containing unique
            IDs for each sample. Defaults to "material_id".
        input_col (str, optional): Column name in train_df and test_df containing input
            values. Defaults to None meaning auto-detect based on 'wren'/'roost' in
            run_name which default to 'protostructure' and 'composition' respectively.
        model_params (dict): Passed to Wrenformer class. E.g. dict(n_attn_layers=6,
            embedding_aggregation=("mean", "std")).
        data_loader_device(str): device to store the InMemoryDataLoader's tensors on.
        **kwargs: Additional keyword arguments are passed to train_model().

    Returns:
        tuple[dict[str, float], dict[str, Any]]: 1st dict are the model's test set
            metrics. 2nd dict are the run's hyperparameters. 3rd is a dataframe with
            test set predictions.
    """
    robust = "robust" in run_name.lower()

    if "wren" in run_name.lower():
        input_col = input_col or "protostructure"
        embedding_type = embedding_type or "protostructure"
    elif "roost" in run_name.lower():
        input_col = input_col or "composition"
        embedding_type = embedding_type or "composition"
    if not input_col or not embedding_type:
        raise ValueError(f"Missing {input_col=} or {embedding_type=} for {run_name=}")

    data_loader_kwargs = dict(
        target_col=target_col,
        input_col=input_col,
        id_col=id_col,
        embedding_type=embedding_type,
        device=data_loader_device,
    )
    train_loader = df_to_in_mem_dataloader(
        train_df,
        batch_size=batch_size,
        shuffle=True,
        **data_loader_kwargs,  # type: ignore[arg-type]
    )

    test_loader = df_to_in_mem_dataloader(
        test_df,
        batch_size=batch_size * inference_multiplier,
        shuffle=False,
        **data_loader_kwargs,  # type: ignore[arg-type]
    )

    # embedding_len is the length of the embedding vector for a Wyckoff position
    # encoding the element type (usually 200-dim matscholar embeddings) and Wyckoff
    # position (see 'bra-alg-off.json') + 1 for the weight of that Wyckoff position (or
    # element) in the material
    embedding_len = train_loader.tensors[0][0].shape[-1]
    # Roost and Wren embedding size resp.
    if embedding_len not in (200 + 1, 200 + 1 + 444):
        raise ValueError(f"{embedding_len=}, expected 201 or 645")

    model_params = dict(
        # 1 for regression, n_classes for classification
        n_targets=[1 if task_type == reg_key else train_df[target_col].max() + 1],
        n_features=embedding_len,
        task_dict={target_col: task_type},  # e.g. {'exfoliation_en': 'regression'}
        robust=robust,
        **model_params or {},
    )
    model = Wrenformer(**model_params)

    test_metrics, run_params, test_df = train_model(
        model=model,
        run_name=run_name,
        target_col=target_col,
        task_type=task_type,
        test_loader=test_loader,
        train_loader=train_loader,
        test_df=test_df,
        model_params=model_params,
        run_params={**kwargs.pop("run_params", {}), **data_loader_kwargs},
        **kwargs,
    )

    return test_metrics, run_params, test_df


def df_train_test_split(
    df: pd.DataFrame,
    folds: tuple[int, int] | None = None,
    test_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets.

    Args:
        df (pd.DataFrame): DataFrame to split
        folds (tuple[int, int] | None, optional): If not None, split the data into
            n_folds[0] folds and use fold with index n_folds[1] as the test set. E.g.
            (10, 0) will create a 90/10 split and use first 10% as the test set.
        test_size (float | None, optional): Fraction of dataframe rows to use as test
            set. Defaults to None.

    Raises:
        ValueError: If folds and test_size are both passed or both None.
            Or if not 0 < test_size < 1 or not 1 < n_folds <= 10.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test sets.
    """
    # shuffle samples for random train/test split
    df_all = df.sample(frac=1, random_state=0)

    if folds:
        n_folds, test_fold_idx = folds
        if not 1 < n_folds <= 10:
            raise ValueError(f"{n_folds = } must be between 2 and 10")
        if not 0 <= test_fold_idx < n_folds:
            raise ValueError(f"{test_fold_idx = } must be between 0 and {n_folds - 1}")

        df_splits: list[pd.DataFrame] = np.array_split(df_all, n_folds)
        test_df = df_splits.pop(test_fold_idx)
        train_df = pd.concat(df_splits)
    elif test_size:
        if not 0 < test_size < 1:
            raise ValueError(f"{test_size = } must be between 0 and 1")

        train_df = df_all.sample(frac=1 - test_size, random_state=0)
        test_df = df_all.drop(train_df.index)
    else:
        raise ValueError(f"Specify either {folds=} or {test_size=}")
    if folds and test_size:
        raise ValueError(f"Specify either {folds=} or {test_size=}, not both")

    return train_df, test_df


@print_walltime(end_desc="predict_from_wandb_checkpoints")
def predict_from_wandb_checkpoints(
    runs: list[wandb.apis.public.Run],
    checkpoint_filename: str = "checkpoint.pth",
    cache_dir: str = "./checkpoint_cache",
    **kwargs: Any,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Download and cache checkpoints for an ensemble of models, then make
    predictions on some dataset. Finally print ensemble metrics and store
    predictions to CSV.

    Args:
        runs (list[wandb.apis.public.Run]): List of WandB runs to download model
            checkpoints from which are then loaded into memory to generate
            predictions for the input_col in df.
        checkpoint_filename (str): Name of the checkpoint file to download.
        cache_dir (str): Directory to cache downloaded checkpoints in.
        **kwargs: Additional keyword arguments to pass to make_ensemble_predictions().

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]: Original input dataframe
            with added columns for model predictions and uncertainties. The optional
            2nd dataframe holds ensemble performance metrics like mean and standard
            deviation of MAE/RMSE.
    """
    print(f"Using checkpoints from {len(runs)} run(s):")

    run_target = runs[0].config["target"]
    if not all(run_target == run.config["target"] for run in runs):
        raise ValueError(f"Runs have differing targets, first {run_target=}")

    target_col = kwargs.get("target_col")
    if target_col and target_col != run_target:
        print(f"\nWarning: {target_col=} does not match {run_target=}")

    checkpoint_paths: list[str] = []

    for idx, run in enumerate(runs, start=1):
        run_path = "/".join(run.path)
        out_dir = f"{cache_dir}/{run_path}"
        os.makedirs(out_dir, exist_ok=True)

        checkpoint_path = f"{out_dir}/{checkpoint_filename}"
        checkpoint_paths.append(checkpoint_path)
        print(f"{idx:>3}/{len(runs)}: {run.url}\n\t{checkpoint_path}\n")

        with open(f"{out_dir}/run.md", "w") as md_file:
            md_file.write(f"[{run.name}]({run.url})\n")

        if not os.path.isfile(checkpoint_path):
            run.file(f"{checkpoint_filename}").download(root=out_dir)

    if target_col is not None:
        df_ens, ensemble_metrics = make_ensemble_predictions(checkpoint_paths, **kwargs)
        # round to save disk space and speed up cloud storage uploads
        return df_ens.round(6), ensemble_metrics

    return make_ensemble_predictions(checkpoint_paths, **kwargs)
