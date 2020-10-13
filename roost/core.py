import gc
import json
import shutil
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from torch.nn.functional import softmax
from tqdm.autonotebook import trange


class BaseModelClass(nn.Module, ABC):
    """
    A base class for models.
    """

    def __init__(self, task, n_targets, robust, device, epoch=1, best_val_score=None):
        """
        Args:
            task (str): "regression" or "classification"
            robust (bool): whether an aleatoric loss function is being used
            device (pytorch.device): the device the model will be run on
            epoch (int): the epoch model training will begin/resume from
            best_val_score (float): validation score to use for early stopping
        """
        super().__init__()
        self.task = task
        self.robust = robust
        self.device = device
        self.epoch = epoch
        self.best_val_score = best_val_score
        self.es_patience = 0

        self.model_params = {}

    def fit(
        self,
        train_generator,
        val_generator,
        optimizer,
        scheduler,
        epochs,
        criterion,
        normalizer,
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
        try:
            for epoch in range(start_epoch, start_epoch + epochs):
                self.epoch += 1
                # Training
                t_loss, t_metrics = self.evaluate(
                    generator=train_generator,
                    criterion=criterion,
                    optimizer=optimizer,
                    normalizer=normalizer,
                    action="train",
                    verbose=verbose,
                )

                if writer is not None:
                    writer.add_scalar("train/loss", t_loss, epoch + 1)
                    for metric, val in t_metrics.items():
                        writer.add_scalar(f"train/{metric}", val, epoch + 1)

                if verbose:
                    print("Epoch: [{}/{}]".format(epoch, start_epoch + epochs - 1))
                    print(
                        f"Train      : Loss {t_loss:.4f}\t"
                        + "".join(
                            [f"{key} {val:.3f}\t" for key, val in t_metrics.items()]
                        )
                    )

                # Validation
                is_best = False
                if val_generator is not None:
                    with torch.no_grad():
                        # evaluate on validation set
                        v_loss, v_metrics = self.evaluate(
                            generator=val_generator,
                            criterion=criterion,
                            optimizer=None,
                            normalizer=normalizer,
                            action="val",
                        )

                    if writer is not None:
                        writer.add_scalar("validation/loss", v_loss, epoch + 1)
                        for metric, val in v_metrics.items():
                            writer.add_scalar(f"validation/{metric}", val, epoch + 1)

                    if verbose:
                        print(
                            f"Validation : Loss {v_loss:.4f}\t"
                            + "".join(
                                [f"{key} {val:.3f}\t" for key, val in v_metrics.items()]
                            )
                        )

                    if self.task == "regression":
                        val_score = v_metrics["MAE"]
                        is_best = val_score < self.best_val_score
                    elif self.task == "classification":
                        val_score = v_metrics["Acc"]
                        is_best = val_score > self.best_val_score

                    if is_best:
                        self.best_val_score = val_score
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
                        "best_val_score": self.best_val_score,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                    if self.task == "regression":
                        checkpoint_dict.update({"normalizer": normalizer.state_dict()})

                    save_checkpoint(checkpoint_dict, is_best, model_name, run_id)

                scheduler.step()

                # catch memory leak
                gc.collect()

        except KeyboardInterrupt:
            pass

        if writer is not None:
            writer.close()

    def evaluate(
        self, generator, criterion, optimizer, normalizer, action="train", verbose=False
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

        loss_meter = AverageMeter()

        if self.task == "regression":
            metric_meter = RegressionMetrics()
        elif self.task == "classification":
            metric_meter = ClassificationMetrics()

        with trange(len(generator), disable=(not verbose)) as t:
            # we do not need batch_comp or batch_ids when training
            for input_, target, _, _ in generator:

                # move tensors to GPU
                input_ = (tensor.to(self.device) for tensor in input_)

                if self.task == "regression":
                    # normalize target if needed
                    target_norm = normalizer.norm(target)
                    target_norm = target_norm.to(self.device)
                elif self.task == "classification":
                    target = target.to(self.device)

                # compute output
                output = self(*input_)

                if self.task == "regression":
                    if self.robust:
                        output, log_std = output.chunk(2, dim=1)
                        loss = criterion(output, log_std, target_norm)
                    else:
                        loss = criterion(output, target_norm)

                    pred = normalizer.denorm(output.data.cpu())
                    metric_meter.update(
                        pred.data.cpu().numpy(), target.data.cpu().numpy()
                    )

                elif self.task == "classification":
                    if self.robust:
                        output, log_std = output.chunk(2, dim=1)
                        logits = sampled_softmax(output, log_std)
                        loss = criterion(torch.log(logits), target.squeeze(1))
                    else:
                        loss = criterion(output, target.squeeze(1))
                        logits = softmax(output, dim=1)

                    # classification metrics from sklearn need numpy arrays
                    metric_meter.update(
                        logits.data.cpu().numpy(), target.data.cpu().numpy()
                    )

                loss_meter.update(loss.data.cpu().item())

                if action == "train":
                    # compute gradient and take an optimizer step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                t.update()

        return loss_meter.avg, metric_meter.metric_dict

    def predict(self, generator, verbose=False):
        """
        evaluate the model
        """

        test_ids = []
        test_comp = []
        test_targets = []
        test_output = []

        # Ensure model is in evaluation mode
        self.eval()

        with torch.no_grad():
            with trange(len(generator), disable=(not verbose)) as t:
                for input_, target, batch_comp, batch_ids in generator:

                    # move tensors to device (GPU or CPU)
                    input_ = (tensor.to(self.device) for tensor in input_)

                    # compute output
                    output = self(*input_)

                    # collect the model outputs
                    test_ids += batch_ids
                    test_comp += batch_comp
                    test_targets.append(target)
                    test_output.append(output)

                    t.update()

        return (
            test_ids,
            test_comp,
            torch.cat(test_targets, dim=0).view(-1).numpy(),
            torch.cat(test_output, dim=0),
        )

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

        with torch.no_grad():
            for input_, *_ in generator:

                input_ = (tensor.to(self.device) for tensor in input_)

                output = self.material_nn(*input_).cpu().numpy()
                features.append(output)

        return np.vstack(features)

    @abstractmethod
    def forward(self, *x):
        """
        Forward pass through the model. Needs to be implemented in any derived
        model class.
        """
        raise NotImplementedError("forward() is not defined!")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RegressionMetrics(object):
    """Computes and stores average metrics for regression tasks"""

    def __init__(self):
        self.rmse_meter = AverageMeter()
        self.mae_meter = AverageMeter()

    def update(self, pred, target):
        mae_error = mae(pred, target)
        self.mae_meter.update(mae_error)

        rmse_error = np.sqrt(mse(pred, target))
        self.rmse_meter.update(rmse_error)

    @property
    def metric_dict(self):
        return {"MAE": self.mae_meter.avg, "RMSE": self.rmse_meter.avg}


class ClassificationMetrics(object):
    """Computes and stores average metrics for classification tasks"""

    def __init__(self):
        self.acc_meter = AverageMeter()
        self.fscore_meter = AverageMeter()

    def update(self, pred, target):
        acc = accuracy_score(target, np.argmax(pred, axis=1))
        self.acc_meter.update(acc)

        fscore = f1_score(target, np.argmax(pred, axis=1), average="weighted")
        self.fscore_meter.update(fscore)

    @property
    def metric_dict(self):
        return {"Acc": self.acc_meter.avg, "F1": self.fscore_meter.avg}


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, log=False):
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


class Featuriser(object):
    """
    Base class for featurising nodes and edges.
    """

    def __init__(self, allowed_types):
        self.allowed_types = set(allowed_types)
        self._embedding = {}

    def get_fea(self, key):
        assert key in self.allowed_types, f"{key} is not an allowed atom type"
        return self._embedding[key]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.allowed_types = set(self._embedding.keys())

    def get_state_dict(self):
        return self._embedding

    @property
    def embedding_size(self):
        return len(self._embedding[list(self._embedding.keys())[0]])


class LoadFeaturiser(Featuriser):
    """
    Initialize a featuriser from a JSON file.

    Parameters
    ----------
    embedding_file: str
        The path to the .json file
    """

    def __init__(self, embedding_file):
        with open(embedding_file) as f:
            embedding = json.load(f)
        allowed_types = set(embedding.keys())
        super().__init__(allowed_types)
        for key, value in embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


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
