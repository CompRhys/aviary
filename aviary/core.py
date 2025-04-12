import gc
import os
import shutil
from abc import ABC
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
import torch
import wandb
from sklearn.metrics import f1_score
from torch import BoolTensor, Tensor, nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from aviary import ROOT
from aviary.data import InMemoryDataLoader, Normalizer

TaskType = Literal["regression", "classification"]


class BaseModelClass(nn.Module, ABC):
    """A base class for models."""

    def __init__(
        self,
        task_dict: dict[str, TaskType],
        robust: bool,
        epoch: int = 0,
        device: str | None = None,
        best_val_scores: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        """Store core model parameters.

        Args:
            task_dict (dict[str, TaskType]): Map target names to "regression" or
                "classification".
            robust (bool): If True, the number of model outputs is doubled. 2nd output
                for each target will be an estimate for the aleatoric uncertainty
                (uncertainty inherent to the sample) which can be used with a robust
                loss function to attenuate the weighting of uncertain samples.
            epoch (int, optional): Epoch model training will begin/resume from.
                Defaults to 0.
            device (str, optional): Device to store the model parameters on.
            best_val_scores (dict[str, float], optional): Validation score to use for
                early stopping. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.task_dict = task_dict
        self.target_names = list(task_dict)
        self.robust = robust
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.epoch = epoch
        self.best_val_scores = best_val_scores or {}
        self.es_patience = 0

        self.to(self.device)
        self.model_params: dict[str, Any] = {"task_dict": task_dict}

    def fit(
        self,
        train_loader: DataLoader | InMemoryDataLoader,
        val_loader: DataLoader | InMemoryDataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epochs: int,
        loss_dict: Mapping[str, tuple[TaskType, Callable]],
        normalizer_dict: Mapping[str, Normalizer | None],
        model_name: str,
        run_id: int,
        checkpoint: bool = True,
        writer: Literal["wandb"] | SummaryWriter | None = None,
        verbose: bool = True,
        patience: int | None = None,
    ) -> None:
        """Ctrl-C interruptible training method.

        Args:
            train_loader (DataLoader): Dataloader containing training data.
            val_loader (DataLoader): Dataloader containing validation data.
            optimizer (torch.optim.Optimizer): Optimizer used to carry out parameter
                updates.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler used to adjust
                Optimizer during training.
            epochs (int): Number of epochs to train for.
            loss_dict (dict[str, nn.Module]): Dict of losses to apply for each task.
            normalizer_dict (dict[str, Normalizer]): Dictionary of Normalizers to apply
                to each task.
            model_name (str): String describing the model.
            run_id (int): Unique identifier of the model run.
            checkpoint (bool, optional): Whether to save model checkpoints.
                Defaults to True.
            writer (SummaryWriter, optional): TensorBoard writer for saving logs.
                Defaults to None.
            verbose (bool, optional): Whether to print out intermediate results.
                Defaults to True.
            patience (int, optional): Patience for early stopping. Defaults to None.
        """
        start_epoch = self.epoch

        try:
            for epoch in range(start_epoch, start_epoch + epochs):
                self.epoch += 1
                # Training
                if verbose:
                    print(f"Epoch: [{epoch}/{start_epoch + epochs - 1}]")
                train_metrics = self.evaluate(
                    train_loader,
                    loss_dict=loss_dict,
                    optimizer=optimizer,
                    normalizer_dict=normalizer_dict,
                    action="train",
                    verbose=verbose,
                )

                if isinstance(writer, SummaryWriter):
                    for task, metrics in train_metrics.items():
                        for metric, val in metrics.items():
                            writer.add_scalar(f"{task}/train/{metric}", val, epoch)

                if writer == "wandb":
                    flat_train_metrics = {}
                    for task, metrics in train_metrics.items():
                        for metric, val in metrics.items():
                            flat_train_metrics[f"train_{task}_{metric.lower()}"] = val
                    flat_train_metrics["epoch"] = epoch
                    wandb.log(flat_train_metrics)

                # Validation
                if val_loader is not None:
                    with torch.no_grad():
                        # evaluate on validation set
                        val_metrics = self.evaluate(
                            val_loader,
                            loss_dict=loss_dict,
                            optimizer=None,
                            normalizer_dict=normalizer_dict,
                            action="evaluate",
                            verbose=verbose,
                        )

                    if isinstance(writer, SummaryWriter):
                        for task, metrics in val_metrics.items():
                            for metric, val in metrics.items():
                                writer.add_scalar(
                                    f"{task}/validation/{metric}", val, epoch
                                )

                    if writer == "wandb":
                        flat_val_metrics = {}
                        for task, metrics in val_metrics.items():
                            for metric, val in metrics.items():
                                flat_val_metrics[f"val_{task}_{metric.lower()}"] = val
                        flat_val_metrics["epoch"] = epoch
                        wandb.log(flat_val_metrics)

                    # TODO test all tasks to see if they are best,
                    # save a best model if any is best.
                    # TODO what are the costs of this approach.
                    # It could involve saving a lot of models?

                    is_best: list[bool] = []

                    for key in self.best_val_scores:
                        score_name = (
                            "MAE" if self.task_dict[key] == "regression" else "Accuracy"
                        )
                        score = val_metrics[key][score_name]
                        prev_best = self.best_val_scores[key]
                        is_best.append(score < prev_best)
                        self.best_val_scores[key] = min(prev_best, score)

                    if any(is_best):
                        self.es_patience = 0
                    else:
                        self.es_patience += 1
                        if patience and self.es_patience > patience:
                            print(
                                f"No improvement on validation set for {patience} "
                                "epochs, stopping early"
                            )
                            break

                if checkpoint:
                    checkpoint_dict = {
                        "model_params": self.model_params,
                        "state_dict": self.state_dict(),
                        "epoch": self.epoch,
                        "best_val_score": self.best_val_scores,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "normalizer_dict": {
                            task: n.state_dict() if isinstance(n, Normalizer) else None
                            for task, n in normalizer_dict.items()
                        },
                    }

                    # TODO saving a model at each epoch may be slow?
                    save_checkpoint(
                        checkpoint_dict,
                        is_best=False,
                        model_name=model_name,
                        run_id=run_id,
                    )

                    # TODO when to save best models? should this be done task-wise in
                    # the multi-task case?
                    # save_checkpoint(checkpoint_dict, is_best, model_name, run_id)

                scheduler.step()

                # catch memory leak
                gc.collect()

        except KeyboardInterrupt:
            pass

        if isinstance(writer, SummaryWriter):
            writer.close()  # close TensorBoard SummaryWriter at end of training

    def evaluate(
        self,
        data_loader: DataLoader | InMemoryDataLoader,
        loss_dict: Mapping[str, tuple[TaskType, nn.Module]],
        optimizer: torch.optim.Optimizer,
        normalizer_dict: Mapping[str, Normalizer | None],
        action: Literal["train", "evaluate"] = "train",
        verbose: bool = False,
        pbar: bool = False,
    ) -> dict[str, dict[str, float]]:
        """Evaluate the model.

        Args:
            data_loader (DataLoader): PyTorch Dataloader with the same data format used
                in fit().
            loss_dict (dict[str, tuple[TaskType, nn.Module]]): Dictionary of losses
                to apply for each task.
            optimizer (torch.optim.Optimizer): PyTorch Optimizer
            normalizer_dict (dict[str, Normalizer]): Dictionary of Normalizers to apply
                to each task.
            action ("train" | "evaluate"], optional): Whether to track gradients
                depending on whether we are carrying out a training or validation pass.
                Defaults to "train".
            verbose (bool, optional): Whether to print out intermediate results.
                Defaults to False.
            pbar (bool, optional): Whether to display a progress bar. Defaults to False.

        Returns:
            dict[str, dict["Loss" | "MAE" | "RMSE" | "Accuracy" | "F1", np.ndarray]]:
                nested dictionary for each target of metrics averaged over an epoch.
        """
        if action == "evaluate":
            self.eval()
        elif action == "train":
            self.train()
        else:
            raise NameError("Only 'train' or 'evaluate' allowed as action")

        epoch_metrics: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # *_ discards identifiers like material_id and formula which we don't need when
        # training tqdm(disable=None) means suppress output in non-tty (e.g. CI/log
        # files) but keep in terminal (i.e. tty mode) https://git.io/JnBOi
        for inputs, targets_list, *_ in tqdm(data_loader, disable=None if pbar else True):
            inputs = [  # noqa: PLW2901
                tensor.to(self.device) if hasattr(tensor, "to") else tensor
                for tensor in inputs
            ]
            outputs = self(*inputs)

            mixed_loss: Tensor = 0  # type: ignore[assignment]

            for target_name, targets, output, normalizer in zip(
                self.target_names,
                targets_list,
                outputs,
                normalizer_dict.values(),
                strict=False,
            ):
                task, loss_func = loss_dict[target_name]
                target_metrics = epoch_metrics[target_name]

                if task == "regression":
                    assert normalizer is not None
                    targets = normalizer.norm(targets).squeeze()  # noqa: PLW2901
                    targets = targets.to(self.device)  # noqa: PLW2901

                    if self.robust:
                        preds, log_std = output.unbind(dim=1)
                        loss = loss_func(preds, log_std, targets)
                    else:
                        preds = output.squeeze(1)
                        loss = loss_func(preds, targets)

                    denormed_preds = normalizer.denorm(preds)
                    denormed_targets = normalizer.denorm(targets)
                    error = denormed_preds - denormed_targets
                    target_metrics["MAE"].append(float(error.abs().mean()))
                    target_metrics["MSE"].append(float(error.pow(2).mean()))

                elif task == "classification":
                    targets = targets.to(self.device)  # noqa: PLW2901

                    if self.robust:
                        pre_logits, log_std = output.chunk(2, dim=1)
                        logits = sampled_softmax(pre_logits, log_std)
                        loss = loss_func(torch.log(logits), targets.squeeze())
                    else:
                        logits = softmax(output, dim=1)
                        loss = loss_func(output, targets)
                    preds = logits

                    logits = logits.data.cpu()
                    targets = targets.data.cpu()  # noqa: PLW2901

                    acc = float((targets == logits.argmax(dim=1)).float().mean())
                    target_metrics["Accuracy"].append(acc)
                    f1 = float(
                        f1_score(targets, logits.argmax(dim=1), average="weighted")
                    )
                    target_metrics["F1"].append(f1)

                else:
                    raise ValueError(f"invalid task: {task}")

                target_metrics["Loss"].append(loss.cpu().item())

                # NOTE multitasking currently just uses a direct sum of individual
                # target losses this should be okay but is perhaps sub-optimal
                mixed_loss += loss

            if action == "train":
                # compute gradient and take an optimizer step
                optimizer.zero_grad()
                mixed_loss.backward()
                optimizer.step()

        avrg_metrics: dict[str, dict[str, float]] = {}
        for target, per_batch_metrics in epoch_metrics.items():
            avrg_metrics[target] = {
                metric_key: np.array(values).mean().squeeze().round(4)
                for metric_key, values in per_batch_metrics.items()
            }
            # take sqrt at the end to get correct epoch RMSE as per-batch averaged RMSE
            # != RMSE of full epoch since (sqrt(a + b) != sqrt(a) + sqrt(b))
            avrg_mse = avrg_metrics[target].pop("MSE", None)
            if avrg_mse:
                avrg_metrics[target]["RMSE"] = (avrg_mse**0.5).round(4)

            if verbose:
                metrics_str = " ".join(
                    f"{key} {val:<9.2f}" for key, val in avrg_metrics[target].items()
                )
                print(f"{action:>9}: {target} N {len(data_loader):,} {metrics_str}")

        return avrg_metrics

    @torch.no_grad()
    def predict(
        self, data_loader: DataLoader | InMemoryDataLoader, verbose: bool = False
    ) -> tuple:
        """Make model predictions. Supports multi-tasking.

        Args:
            data_loader (DataLoader): Iterator that yields mini-batches with the same
                data format used in fit(). To speed up inference, batch size can be set
                much larger than during training.
            verbose (bool, optional): Whether to print out intermediate results.
                Defaults to False.

        Returns:
            3 tuples where tuple items correspond to different multitask targets.
            - tuple[np.array, ...]: Tuple of target Tensors
            - tuple[np.array, ...]: Tuple of prediction Tensors
            - tuple[list[str], ...]: Tuple of identifiers
        If single task, tuple will have length 1. Use this code to unpack:
        targets, preds, ids = model.predict(data_loader)
        targets, preds, ids = targets[0], preds[0], ids[0]
        """
        test_ids = []
        test_targets = []
        test_preds = []
        # Ensure model is in evaluation mode
        self.eval()

        # disable output in non-tty (e.g. log files) https://git.io/JnBOi
        for inputs, targets, *batch_ids in tqdm(
            data_loader, disable=True if not verbose else None
        ):
            inputs = [  # noqa: PLW2901
                tensor.to(self.device) if hasattr(tensor, "to") else tensor
                for tensor in inputs
            ]
            preds = self(*inputs)  # forward pass to get model preds

            test_ids.append(batch_ids)
            test_targets.append(targets)
            test_preds.append(preds)

        # NOTE zip(*...) transposes list dims 0 (n_batches) and 1 (n_tasks)
        # for multitask learning
        targets = tuple(
            torch.cat(targets, dim=0).view(-1).cpu().numpy()
            for targets in zip(*test_targets, strict=False)
        )
        predictions = tuple(
            torch.cat(preds, dim=0) for preds in zip(*test_preds, strict=False)
        )
        # identifier columns
        ids = tuple(np.concatenate(x) for x in zip(*test_ids, strict=False))
        return targets, predictions, ids

    @torch.no_grad()
    def featurize(self, data_loader: DataLoader) -> np.ndarray:
        """Generate features for a list of composition strings. When using Roost,
        this runs only the message-passing part of the model without the ResNet.

        Args:
            data_loader (DataLoader): PyTorch Dataloader with the same data format used
                in fit()

        Returns:
            np.array: 2d array of features
        """
        if self.epoch <= 0:
            raise AssertionError(
                f"{self} needs to be fitted before it can be used for featurization"
            )

        self.eval()  # ensure model is in evaluation mode
        features = []

        for inputs, *_ in data_loader:
            inputs = [  # noqa: PLW2901
                tensor.to(self.device) if hasattr(tensor, "to") else tensor
                for tensor in inputs
            ]
            output = self.trunk_nn(self.material_nn(*inputs)).cpu().numpy()
            features.append(output)

        return np.vstack(features)

    @property
    def num_params(self) -> int:
        """Return number of trainable parameters in model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """Return model name with number of parameters and epochs trained."""
        n_params, n_epochs = self.num_params, self.epoch
        cls_name = type(self).__name__
        return f"{cls_name} with {n_params:,} trainable params at {n_epochs:,} epochs"


def save_checkpoint(
    state: dict[str, Any], is_best: bool, model_name: str, run_id: int
) -> None:
    """Saves a checkpoint and overwrites the best model when is_best = True.

    Args:
        state (dict[str, Any]): Model parameters and other stateful objects like
            optimizer.
        is_best (bool): Whether the model is the best seen according to validation set.
        model_name (str): String describing the model.
        run_id (int): Unique identifier of the model run.
    """
    model_dir = f"{ROOT}/models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    checkpoint = f"{model_dir}/checkpoint-r{run_id}.pth.tar"
    best = f"{model_dir}/best-r{run_id}.pth.tar"

    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)


def sampled_softmax(pre_logits: Tensor, log_std: Tensor, samples: int = 10) -> Tensor:
    """Draw samples from Gaussian distributed pre-logits and use these to estimate
    a mean and aleatoric uncertainty.

    Args:
        pre_logits (Tensor): Expected logits before softmax.
        log_std (Tensor): Deviation in logits before softmax.
        samples (int, optional): Number of samples to take. Defaults to 10.

    Returns:
        Tensor: Averaged logits sampled from pre-logits
    """
    # NOTE here as we do not risk dividing by zero should we really be
    # predicting log_std or is there another way to deal with negative numbers?
    # This choice may have an unknown effect on the calibration of the uncertainties
    sam_std = torch.exp(log_std).repeat_interleave(samples, dim=0)

    # TODO here we are normally distributing the samples even if the loss
    # uses a different prior?
    epsilon = torch.randn_like(sam_std)
    pre_logits = pre_logits.repeat_interleave(samples, dim=0) + torch.mul(
        epsilon, sam_std
    )
    logits = softmax(pre_logits, dim=1).view(len(log_std), samples, -1)
    return torch.mean(logits, dim=1)


def np_softmax(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute the softmax of an array along an axis.

    Args:
        arr (np.ndarray): Arbitrary dimensional array.
        axis (int, optional): Dimension over which to take softmax. Defaults to
            -1 (last).

    Returns:
        np.ndarray: Same dimension as input array, but specified axis reduced
            to singleton.
    """
    exp = np.exp(arr)
    return exp / exp.sum(axis=axis, keepdims=True)


def np_one_hot(targets: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Get a one-hot encoded version of an array of class labels.

    Args:
        targets (np.ndarray): 1d array of integer class labels.
        n_classes (int, optional): Number of classes. Defaults to np.max(targets) + 1.

    Returns:
        np.ndarray: 2d array of 1-hot encoded class labels.
    """
    if n_classes is None:
        n_classes = np.max(targets) + 1
    return np.eye(n_classes)[targets]


def masked_std(x: Tensor, mask: BoolTensor, dim: int = 0, eps: float = 1e-12) -> Tensor:
    """Compute the standard deviation of a tensor, ignoring masked values.

    Args:
        x (Tensor): Tensor to compute standard deviation of.
        mask (BoolTensor): Same shape as x with True where x is valid and False
            where x should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs.
        dim (int, optional): Dimension to take std of. Defaults to 0.
        eps (float, optional): Small positive number to ensure std is differentiable.
            Defaults to 1e-12.

    Returns:
        Tensor: Same shape as x, except dimension dim reduced.
    """
    mean = masked_mean(x, mask, dim=dim)
    squared_diff = (x - mean.unsqueeze(dim=dim)) ** 2
    var = masked_mean(squared_diff, mask, dim=dim)
    return (var + eps).sqrt()


def masked_mean(x: Tensor, mask: BoolTensor, dim: int = 0) -> Tensor:
    """Compute the mean of a tensor, ignoring masked values.

    Args:
        x (Tensor): Tensor to compute mean of.
        mask (BoolTensor): Same shape as x with True where x is valid and False
            where x should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.
        dim (int, optional): Dimension to take mean of. Defaults to 0.

    Returns:
        Tensor: Same shape as x, except dimension dim reduced.
    """
    # for safety, we could add this assert but might impact performance
    # assert (
    #     mask.sum(dim=dim).ne(0).all()
    # ), "mask should not be all False in any column, causes zero division"
    x_nan = x.float().masked_fill(~mask, float("nan"))
    return x_nan.nanmean(dim=dim)


def masked_max(x: Tensor, mask: BoolTensor, dim: int = 0) -> Tensor:
    """Compute the max of a tensor along dimension dim, ignoring values at indices where
    mask is False. See masked_mean docstring for Args details.
    """
    # replace padded values with +/-inf to make sure min()/max() ignore them
    x_inf = x.float().masked_fill(~mask, float("-inf"))
    # 1st ret val = max, 2nd ret val = max indices
    x_max, _ = x_inf.max(dim=dim)
    return x_max


def masked_min(x: Tensor, mask: BoolTensor, dim: int = 0) -> Tensor:
    """Compute the min of a tensor along dimension dim, ignoring values at indices where
    mask is False. See masked_mean docstring for Args details.
    """
    x_inf = x.float().masked_fill(~mask, float("inf"))
    x_min, _ = x_inf.min(dim=dim)
    return x_min


AGGREGATORS: dict[str, Callable[[Tensor, BoolTensor, int], Tensor]] = {
    "mean": masked_mean,
    "std": masked_std,
    "max": masked_max,
    "min": masked_min,
    "sum": lambda x, mask, dim: (x * mask).sum(dim=dim),
}
