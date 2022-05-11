from __future__ import annotations

import gc
import os
import shutil
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from aviary import ROOT
from aviary.data import InMemoryDataLoader

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


TaskType = Literal["regression", "classification"]


class BaseModelClass(nn.Module, ABC):
    """A base class for models."""

    def __init__(
        self,
        task_dict: dict[str, TaskType],
        robust: bool,
        epoch: int = 0,
        best_val_scores: dict[str, float] = None,
    ) -> None:
        """Store core model parameters.

        Args:
            task_dict (dict[str, TaskType]): Map target names to "regression" or "classification".
            robust (bool): Whether to estimate standard deviation for use in a robust loss function
            epoch (int, optional): Epoch model training will begin/resume from. Defaults to 0.
            best_val_scores (dict[str, float], optional): Validation score to use for early
                stopping. Defaults to None.
        """
        super().__init__()
        self.task_dict = task_dict
        self.target_names = list(task_dict.keys())
        self.robust = robust
        self.epoch = epoch
        self.best_val_scores = best_val_scores or {}
        self.es_patience = 0

        self.model_params = {"task_dict": task_dict}

    def fit(  # noqa: C901
        self,
        train_generator: DataLoader | InMemoryDataLoader,
        val_generator: DataLoader | InMemoryDataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epochs: int,
        criterion_dict: dict[str, tuple[TaskType, nn.Module]],
        normalizer_dict: dict[str, Normalizer | None],
        model_name: str,
        run_id: int,
        checkpoint: bool = True,
        writer: SummaryWriter = None,
        verbose: bool = True,
        patience: int = None,
    ) -> None:
        """Ctrl-C interruptible training method.

        Args:
            train_generator (DataLoader): Dataloader containing training data.
            val_generator (DataLoader): Dataloader containing validation data.
            optimizer (torch.optim.Optimizer): Optimizer used to carry out parameter updates.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler used to adjust
                Optimizer during training.
            epochs (int): Number of epochs to train for.
            criterion_dict (dict[str, nn.Module]): Dictionary of losses to apply for each task.
            normalizer_dict (dict[str, Normalizer]): Dictionary of Normalizers to apply
                to each task.
            model_name (str): String describing the model.
            run_id (int): Unique identifier of the model run.
            checkpoint (bool, optional): Whether to save model checkpoints. Defaults to True.
            writer (SummaryWriter, optional): TensorBoard writer for saving logs. Defaults to None.
            verbose (bool, optional): Whether to print out intermediate results. Defaults to True.
            patience (int, optional): Patience for early stopping. Defaults to None.
        """
        start_epoch = self.epoch

        try:
            for epoch in range(start_epoch, start_epoch + epochs):
                self.epoch += 1
                # Training
                t_metrics = self.evaluate(
                    generator=train_generator,
                    criterion_dict=criterion_dict,
                    optimizer=optimizer,
                    normalizer_dict=normalizer_dict,
                    action="train",
                    verbose=verbose,
                )

                if writer is not None:
                    for task, metrics in t_metrics.items():
                        for metric, val in metrics.items():
                            writer.add_scalar(f"{task}/train/{metric}", val, epoch)

                if verbose:
                    print(f"Epoch: [{epoch}/{start_epoch + epochs - 1}]")
                    for task, metrics in t_metrics.items():
                        metrics_str = "".join(
                            [f"{key} {val:.2f}\t" for key, val in metrics.items()]
                        )
                        print(f"Train \t\t: {task} - {metrics_str}")

                # Validation
                if val_generator is not None:
                    with torch.no_grad():
                        # evaluate on validation set
                        v_metrics = self.evaluate(
                            generator=val_generator,
                            criterion_dict=criterion_dict,
                            optimizer=None,
                            normalizer_dict=normalizer_dict,
                            action="val",
                        )

                    if writer is not None:
                        for task, metrics in v_metrics.items():
                            for metric, val in metrics.items():
                                writer.add_scalar(
                                    f"{task}/validation/{metric}", val, epoch
                                )

                    if verbose:
                        for task, metrics in v_metrics.items():
                            metrics_str = "".join(
                                [f"{key} {val:.2f}\t" for key, val in metrics.items()]
                            )
                            print(f"Validation \t: {task} - {metrics_str}")

                    # TODO test all tasks to see if they are best,
                    # save a best model if any is best.
                    # TODO what are the costs of this approach.
                    # It could involve saving a lot of models?

                    is_best: list[bool] = []

                    for name in self.best_val_scores:
                        if self.task_dict[name] == "regression":
                            if v_metrics[name]["MAE"] < self.best_val_scores[name]:
                                self.best_val_scores[name] = v_metrics[name]["MAE"]
                                is_best.append(True)
                            is_best.append(False)
                        elif self.task_dict[name] == "classification":
                            if v_metrics[name]["Acc"] > self.best_val_scores[name]:
                                self.best_val_scores[name] = v_metrics[name]["Acc"]
                                is_best.append(True)
                            is_best.append(False)

                    if any(is_best):
                        self.es_patience = 0
                    else:
                        self.es_patience += 1
                        if patience and self.es_patience > patience:
                            print(
                                "Stopping early due to lack of improvement on Validation set"
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
                    save_checkpoint(checkpoint_dict, False, model_name, run_id)

                    # TODO when to save best models? should this be done task-wise in
                    # the multi-task case?
                    # save_checkpoint(checkpoint_dict, is_best, model_name, run_id)

                scheduler.step()

                # catch memory leak
                gc.collect()

        except KeyboardInterrupt:
            pass

        if writer is not None:
            writer.close()

    def evaluate(
        self,
        generator: DataLoader,
        criterion_dict: dict[str, tuple[TaskType, nn.Module]],
        optimizer: torch.optim.Optimizer,
        normalizer_dict: dict[str, Normalizer | None],
        action: Literal["train", "val"] = "train",
        verbose: bool = False,
    ):
        """Evaluate the model.

        Args:
            generator (DataLoader): PyTorch Dataloader with the same data format used in fit().
            criterion_dict (dict[str, tuple[TaskType, nn.Module]]): Dictionary of losses
                to apply for each task.
            optimizer (torch.optim.Optimizer): PyTorch Optimizer
            normalizer_dict (dict[str, Normalizer]): Dictionary of Normalizers to apply
                to each task.
            action ("train" | "val"], optional): Whether to track gradients depending on
                whether we are carrying out a training or validation pass. Defaults to "train".
            verbose (bool, optional): Whether to print out intermediate results. Defaults to False.

        Returns:
            dict[str, dict["Loss" | "MAE" | "RMSE" | "Acc" | "F1", np.ndarray]]: nested
                dictionary of metrics for each task.
        """
        if action == "val":
            self.eval()
        elif action == "train":
            self.train()
        else:
            raise NameError("Only train or val allowed as action")

        metrics: dict[str, dict[Literal["Loss", "MAE", "RMSE", "Acc", "F1"], list]] = {
            key: defaultdict(list) for key in self.task_dict
        }

        # we do not need batch_comp or batch_ids when training
        # disable output in non-tty (e.g. log files) https://git.io/JnBOi
        for inputs, targets, *_ in tqdm(
            generator, disable=True if not verbose else None
        ):
            normed_targets = [
                n.norm(tar) if n is not None else tar
                for tar, n in zip(targets, normalizer_dict.values())
            ]

            # compute output
            outputs = self(*inputs)

            mixed_loss: Tensor = 0

            for name, output, target in zip(self.target_names, outputs, normed_targets):
                task, criterion = criterion_dict[name]

                if task == "regression":
                    if self.robust:
                        output, log_std = output.chunk(2, dim=1)
                        output = output.squeeze()
                        loss = criterion(output, log_std, target)
                    else:
                        output = output.squeeze()
                        loss = criterion(output, target)

                    pred = normalizer_dict[name].denorm(output.data.cpu())  # type: ignore
                    target = normalizer_dict[name].denorm(target.data.cpu())  # type: ignore
                    metrics[name]["MAE"].append((pred - target).abs().mean())
                    metrics[name]["RMSE"].append((pred - target).pow(2).mean().sqrt())

                elif task == "classification":
                    if self.robust:
                        output, log_std = output.chunk(2, dim=1)
                        logits = sampled_softmax(output, log_std)
                        loss = criterion(torch.log(logits), target.squeeze(1))
                    else:
                        logits = softmax(output, dim=1)
                        loss = criterion(output, target.squeeze())

                    logits = logits.data.cpu()
                    target = target.squeeze().data.cpu()

                    # classification metrics from sklearn need numpy arrays
                    metrics[name]["Acc"].append(
                        accuracy_score(target, np.argmax(logits, axis=1))
                    )
                    metrics[name]["F1"].append(
                        f1_score(target, np.argmax(logits, axis=1), average="weighted")
                    )
                else:
                    raise ValueError(f"invalid task: {task}")

                metrics[name]["Loss"].append(loss.cpu().item())

                # NOTE we are currently just using a direct sum of losses
                # this should be okay but is perhaps sub-optimal
                mixed_loss += loss

            if action == "train":
                # compute gradient and take an optimizer step
                optimizer.zero_grad()
                mixed_loss.backward()
                optimizer.step()

        metrics = {
            key: {k: np.array(v).mean() for k, v in d.items() if v}
            for key, d in metrics.items()
        }

        return metrics

    @torch.no_grad()
    def predict(
        self, generator: DataLoader | InMemoryDataLoader, verbose: bool = False
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[str, ...]]:
        """Make model predictions.

        Args:
            generator (DataLoader): PyTorch Dataloader with the same data format used in fit()
            verbose (bool, optional): Whether to print out intermediate results. Defaults to False.

        Returns:
            3-tuple containing:
            - tuple[Tensor, ...]: Tuple of target Tensors
            - tuple[Tensor, ...]: Tuple of prediction Tensors
            - tuple[str, ...]: Tuple of identifiers
        """
        test_ids = []
        test_targets = []
        test_outputs = []
        # Ensure model is in evaluation mode
        self.eval()

        # disable output in non-tty (e.g. log files) https://git.io/JnBOi
        for input_, targets, *batch_ids in tqdm(
            generator, disable=True if not verbose else None
        ):
            # compute output
            output = self(*input_)

            # collect the model outputs

            test_ids.append(batch_ids)
            test_targets.append(targets)
            test_outputs.append(output)

        # NOTE zip(*...) transposes list dims 0 (n_batches) and 1 (n_tasks)
        # for multitask learning
        targets = tuple(
            torch.cat(test_t, dim=0).view(-1).cpu().numpy()
            for test_t in zip(*test_targets)
        )
        predictions = tuple(torch.cat(test_o, dim=0) for test_o in zip(*test_outputs))
        # identifier columns
        ids = [list(chain(*x)) for x in list(zip(*test_ids))]
        return targets, predictions, *ids  # type: ignore

    @torch.no_grad()
    def featurise(self, generator: DataLoader) -> np.ndarray:
        """Generate features for a list of composition strings. When using Roost,
        this runs only the message-passing part of the model without the ResNet.

        Args:
            generator (DataLoader): PyTorch Dataloader with the same data format used in fit()

        Returns:
            np.array: 2d array of features
        """
        err_msg = f"{self} needs to be fitted before it can be used for featurisation"
        if self.epoch <= 0:
            raise AssertionError(err_msg)

        self.eval()  # ensure model is in evaluation mode
        features = []

        for input_, *_ in generator:
            output = self.trunk_nn(self.material_nn(*input_)).cpu().numpy()
            features.append(output)

        return np.vstack(features)

    @abstractmethod
    def forward(self, *x):
        """Forward pass through the model.

        Raises:
            NotImplementedError: Raise error if child class doesn't implement forward
        """
        raise NotImplementedError("forward() is not defined!")

    @property
    def num_params(self) -> int:
        """Return number of trainable parameters in model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """Return unambiguous string representation of model."""
        name = self._get_name()
        n_params, n_epochs = self.num_params, self.epoch
        return f"{name} with {n_params:,} trainable params at {n_epochs:,} epochs"


class Normalizer:
    """Normalize a Tensor and restore it later."""

    def __init__(self) -> None:
        self.mean = torch.tensor(0)
        self.std = torch.tensor(1)

    def fit(self, tensor: Tensor, dim: int = 0, keepdim: bool = False) -> None:
        """Compute the mean and standard deviation of the given tensor.

        Args:
            tensor (Tensor): Tensor to determine the mean and standard deviation over.
            dim (int, optional): Which dimension to take mean and standard deviation over.
                Defaults to 0.
            keepdim (bool, optional): Whether to keep the reduced dimension in Tensor.
                Defaults to False.
        """
        self.mean = torch.mean(tensor, dim, keepdim)
        self.std = torch.std(tensor, dim, keepdim)

    def norm(self, tensor: Tensor) -> Tensor:
        """Normalize a Tensor

        Args:
            tensor (Tensor): Tensor to be normalized

        Returns:
            Tensor: Normalized Tensor
        """
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: Tensor) -> Tensor:
        """Restore normalized Tensor to original.

        Args:
            tensor (Tensor): Tensor to be restored

        Returns:
            Tensor: Restored Tensor
        """
        return normed_tensor * self.std + self.mean

    def state_dict(self) -> dict[str, Tensor]:
        """Dictionary storing Normalizer parameters

        Returns:
            dict[str, Tensor]: Dictionary storing Normalizer parameters.
        """
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        """Overwrite Normalizer parameters given a new state_dict

        Args:
            state_dict (dict[str, Tensor]): Dictionary storing Normalizer parameters.
        """
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, Tensor]) -> Normalizer:
        """Create a new Normalizer given a state_dict

        Args:
            state_dict (dict[str, Tensor]): Dictionary storing Normalizer parameters.

        Returns:
            Normalizer
        """
        instance = cls()
        instance.mean = state_dict["mean"].cpu()
        instance.std = state_dict["std"].cpu()

        return instance


def save_checkpoint(
    state: dict[str, Any], is_best: bool, model_name: str, run_id: int
) -> None:
    """Saves a checkpoint and overwrites the best model when is_best = True.

    Args:
        state (dict[str, Any]): Dictionary containing model parameters.
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
