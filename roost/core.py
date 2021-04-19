import gc
import json
import shutil
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import softmax
from tqdm.autonotebook import tqdm


class BaseModelClass(nn.Module, ABC):
    """
    A base class for models.
    """

    def __init__(
        self,
        task_dict,
        robust,
        device,
        epoch=1,
        best_val_scores=None
    ):
        """
        Args:
            task (str): "regression" or "classification"
            robust (bool): whether an aleatoric loss function is being used
            device (pytorch.device): the device the model will be run on
            epoch (int): the epoch model training will begin/resume from
            best_val_score (float): validation score to use for early stopping
        """
        super().__init__()
        self.task_dict = task_dict
        self.target_names = list(task_dict.keys())
        self.robust = robust
        self.device = device
        self.epoch = epoch
        self.best_val_scores = best_val_scores
        self.es_patience = 0

        self.model_params = {"task_dict": task_dict}

    def fit(
        self,
        train_generator,
        val_generator,
        optimizer,
        scheduler,
        epochs,
        criterion_dict,
        normalizer_dict,
        model_name,
        run_id,
        checkpoint=True,
        writer=None,
        verbose=True,
        patience=None,
    ):
        """
        Args:

        """
        start_epoch = self.epoch

        # if writer is not None:
        #     for name, param in self.named_parameters():
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), start_epoch)

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

                        # if epoch % 5 == 0:
                        #     for name, param in self.named_parameters():
                        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                        #         writer.add_histogram(name+"/grad", param.grad.clone().cpu().data.numpy(), epoch)

                if verbose:
                    print("Epoch: [{}/{}]".format(epoch, start_epoch + epochs - 1))
                    for task, metrics in t_metrics.items():
                        print(
                            f"Train \t\t: {task} - "
                            + "".join(
                                [f"{key} {val:.3f}\t" for key, val in metrics.items()]
                            )
                        )

                # Validation
                is_best = False
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
                                writer.add_scalar(f"{task}/validation/{metric}", val, epoch)

                    if verbose:
                        for task, metrics in v_metrics.items():
                            print(
                                f"Validation \t: {task} - "
                                + "".join(
                                    [f"{key} {val:.3f}\t" for key, val in metrics.items()]
                                )
                            )

                    # TODO test all tasks to see if they are best, save a best model if any is best.
                    # TODO what are the costs of this approach is could involve saving a lot of models?

                    is_best = []

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
                        if patience:
                            if self.es_patience > patience:
                                print("Stopping early due to lack of improvement on Validation set")
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
                            task: n.state_dict() if isinstance(n, Normalizer)
                            else None
                            for task, n in normalizer_dict.items()
                        }
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
        generator,
        criterion_dict,
        optimizer,
        normalizer_dict,
        action="train",
        verbose=False
    ):
        """
        evaluate the model
        """

        if action == "val":
            self.eval()
        elif action == "train":
            self.train()
        else:
            raise NameError("Only train or val allowed as action")

        metrics = {
            key: {
                k: [] for k in ["Loss", "MAE", "RMSE", "Acc", "F1"]
            } for key in self.task_dict
        }

        # we do not need batch_comp or batch_ids when training
        for inputs, targets, *_ in tqdm(generator, disable=not verbose):

            # move tensors to GPU
            inputs = (tensor.to(self.device) for tensor in inputs)

            targets = (n.norm(tar) if n is not None else tar
                        for tar, n in zip(targets, normalizer_dict.values()))

            targets = [target.to(self.device) for target in targets]

            # compute output
            outputs = self(*inputs)

            mixed_loss = 0

            for name, output, target in zip(self.target_names, outputs, targets):
                task, criterion = criterion_dict[name]

                if task == "regression":
                    if self.robust:
                        output, log_std = output.chunk(2, dim=1)
                        loss = criterion(output, log_std, target)
                    else:
                        loss = criterion(output, target)

                    pred = normalizer_dict[name].denorm(output.data.cpu())
                    target = normalizer_dict[name].denorm(target.data.cpu())
                    metrics[name]['MAE'].append((pred - target).abs().mean())
                    metrics[name]['RMSE'].append((pred - target).pow(2).mean().sqrt())

                elif task == "classification":
                    if self.robust:
                        output, log_std = output.chunk(2, dim=1)
                        logits = sampled_softmax(output, log_std)
                        loss = criterion(torch.log(logits), target.squeeze(1))
                    else:
                        logits = softmax(output, dim=1)
                        loss = criterion(output, target.squeeze(1))

                    logits = logits.data.cpu()
                    target = target.squeeze(1).data.cpu()

                    # classification metrics from sklearn need numpy arrays
                    metrics[name]['Acc'].append(
                        accuracy_score(target, np.argmax(logits, axis=1))
                    )
                    metrics[name]['F1'].append(
                        f1_score(target, np.argmax(logits, axis=1), average="weighted")
                    )

                elif task == "mask":
                    logits = softmax(output, dim=-1)
                    loss = criterion(logits, target)

                    logits = logits.data.cpu()
                    target = target.data.cpu()

                    # classification metrics from sklearn need numpy arrays
                    # NOTE these metrics are misleading on disordered structures
                    metrics[name]['Acc'].append(
                        accuracy_score(np.argmax(target, axis=1), np.argmax(logits, axis=1))
                    )
                    metrics[name]['F1'].append(
                        f1_score(np.argmax(target, axis=1), np.argmax(logits, axis=1), average="weighted")
                    )

                else:
                    raise ValueError(f"invalid task: {task}")

                metrics[name]['Loss'].append(loss.cpu().item())

                # NOTE we are currently just using a direct sum of losses
                # this should be okay but is perhaps sub-optimal
                mixed_loss += loss

            if action == "train":
                # compute gradient and take an optimizer step
                optimizer.zero_grad()
                mixed_loss.backward()
                optimizer.step()

        metrics = {key: {k: np.array(v).mean() for k, v in d.items() if v} for key, d in metrics.items()}

        return metrics

    @torch.no_grad()
    def predict(self, generator, verbose=False):
        """
        evaluate the model
        """

        test_ids = []
        test_comp = []
        test_targets = []
        test_outputs = []
        # Ensure model is in evaluation mode
        self.eval()

        for input_, targets, batch_comp, batch_ids in tqdm(generator, disable=not verbose):

            # move tensors to device (GPU or CPU)
            input_ = (tensor.to(self.device) for tensor in input_)

            # compute output
            output = self(*input_)

            # collect the model outputs
            test_ids += batch_ids
            test_comp += batch_comp
            test_targets.append(targets)
            test_outputs.append(output)

        return (
            test_ids,
            test_comp,
            # NOTE zip(*...) transposes list dims 0 (n_batches) and 1 (n_tasks)
            # for multitask learning
            (torch.cat(test_t, dim=0).view(-1).numpy() for test_t in zip(*test_targets)),
            (torch.cat(test_o, dim=0) for test_o in zip(*test_outputs)),
        )

    @torch.no_grad()
    def featurise(self, generator):
        """Generate features for a list of composition strings. When using Roost,
        this runs only the message-passing part of the model without the ResNet.

        Args:
            generator (DataLoader): PyTorch loader with the same data format used in fit()

        Returns:
            np.array: 2d array of features
        """
        err_msg = f"{self} needs to be fitted before it can be used for featurisation"
        assert self.epoch > 0, err_msg

        self.eval()  # ensure model is in evaluation mode
        features = []

        for input_, *_ in generator:

            input_ = (tensor.to(self.device) for tensor in input_)

            output = self.trunk_nn(self.material_nn(*input_)).cpu().numpy()
            features.append(output)

        return np.vstack(features)

    @abstractmethod
    def forward(self, *x):
        """
        Forward pass through the model. Needs to be implemented in any derived
        model class.
        """
        raise NotImplementedError("forward() is not defined!")

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.tensor((0))
        self.std = torch.tensor((1))

    def fit(self, tensor, dim=0, keepdim=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor, dim, keepdim)
        self.std = torch.std(tensor, dim, keepdim)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()

    @classmethod
    def from_state_dict(cls, state_dict):
        instance = cls()
        instance.mean = state_dict["mean"].cpu()
        instance.std = state_dict["std"].cpu()

        return instance


class Featurizer:
    """Base class for featurizing nodes and edges."""

    def __init__(self, allowed_types):
        self.allowed_types = allowed_types
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types, f"{key} is not an allowed atom type"
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = self._embedding.keys()

    def get_state_dict(self):
        return self._embedding

    @property
    def embedding_size(self):
        return len(list(self._embedding.values())[0])

    @classmethod
    def from_json(cls, embedding_file):
        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = embedding.keys()
        instance = cls(allowed_types)
        for key, value in embedding.items():
            instance._embedding[key] = np.array(value, dtype=float)
        return instance


def save_checkpoint(state, is_best, model_name, run_id):
    """
    Saves a checkpoint and overwrites the best model when is_best = True
    """
    checkpoint = f"models/{model_name}/checkpoint-r{run_id}.pth.tar"
    best = f"models/{model_name}/best-r{run_id}.pth.tar"

    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)


def RobustL1Loss(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = np.sqrt(2.0) * torch.abs(output - target) * torch.exp(-log_std) + log_std
    return torch.mean(loss)


def RobustL2Loss(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    # NOTE can we scale log_std by something sensible to improve the OOD behaviour?
    loss = 0.5 * torch.pow(output - target, 2.0) * torch.exp(-2.0 * log_std) + log_std
    return torch.mean(loss)


def sampled_softmax(pre_logits, log_std, samples=10):
    """
    Draw samples from gaussian distributed pre-logits and use these to estimate
    a mean and aleatoric uncertainty.
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
