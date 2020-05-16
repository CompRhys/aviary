import os
import gc
import torch
import shutil
import numpy as np
import torch.nn as nn

from tqdm.autonotebook import trange
from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse
from torch.nn.functional import softmax

from sklearn.metrics import accuracy_score, f1_score

class BaseModelClass(nn.Module):
    """
    A base class for models.
    """

    def __init__(self, task, n_targets, robust, device, epoch=1, best_val_score=None):
        super(BaseModelClass, self).__init__()
        self.task = task
        self.n_targets = n_targets
        self.robust = robust
        if self.task == "regression":
            self.scoring_rule = "MAE"
        elif self.task == "classification":
            self.scoring_rule = "Acc"
        self.device = device
        self.epoch = epoch
        self.best_val_score = best_val_score

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
        writer=None,
    ):
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
                    verbose=True,
                )

                print("Epoch: [{}/{}]".format(epoch, start_epoch + epochs - 1))
                print(f"Train      : Loss {t_loss:.4f}\t"
                    + "".join([f"{key} {val:.3f}\t" for key, val in t_metrics.items()])
                )

                # Validation
                if val_generator is None:
                    is_best = False
                else:
                    with torch.no_grad():
                        # evaluate on validation set
                        v_loss, v_metrics = self.evaluate(
                            generator=val_generator,
                            criterion=criterion,
                            optimizer=None,
                            normalizer=normalizer,
                            action="val",
                        )

                    print(f"Validation : Loss {v_loss:.4f}\t"
                        + "".join([f"{key} {val:.3f}\t" for key, val in v_metrics.items()])       
                    )

                    # NOTE we need to find a proper scoring rule for classification 
                    # which is minimised for good models.
                    is_best = v_metrics[self.scoring_rule] < self.best_val_score

                if is_best:
                    self.best_val_score = v_metrics[self.scoring_rule]

                checkpoint_dict = {
                    "state_dict": self.state_dict(),
                    "epoch": self.epoch,
                    "best_val_score": self.best_val_score,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                if self.task == "regression":
                    checkpoint_dict.update({"normalizer": normalizer.state_dict()})

                save_checkpoint(checkpoint_dict, is_best, model_name, run_id)

                if writer is not None:
                    writer.add_scalar("train/loss", t_loss, epoch + 1)
                    for metric, val in t_metrics.items():
                        writer.add_scalar(f"train/{metric}", val, epoch + 1)

                    if val_generator is not None:
                        writer.add_scalar("validation/loss", v_loss, epoch + 1)
                        for metric, val in v_metrics.items():
                            writer.add_scalar(f"validation/{metric}", val, epoch + 1)

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
            for input_, target, batch_comp, batch_ids in generator:

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
                    metric_meter.update(pred, target)

                elif self.task == "classification":
                    if self.robust:
                        output, log_std = output.chunk(2, dim=1)  
                        logits = sampled_softmax(output, log_std)
                        loss = criterion(torch.log(logits), target.squeeze(1))
                    else:
                        loss = criterion(output, target.squeeze(1))
                        logits = softmax(output, dim=1)

                    # classification metrics from sklearn need numpy arrays
                    metric_meter.update(logits.data.cpu().numpy(), target.data.cpu().numpy())

                loss_meter.update(loss.data.cpu().item())

                if action == "train":
                    # compute gradient and take an optimizer step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                t.update()

        return loss_meter.avg, metric_meter.metric_dict()

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

        with trange(len(generator), disable=(not verbose)) as t:
            for input_, target, batch_comp, batch_ids in generator:

                # move tensors to GPU
                input_ = (tensor.to(self.device) for tensor in input_)

                # compute output
                output = self(*input_)

                # collect the model outputs
                test_ids += batch_ids
                test_comp += batch_comp
                test_targets.append(target)
                test_output.append(output)

                t.update()

        return test_ids, test_comp, torch.cat(test_targets, dim=0).view(-1).numpy(), \
                torch.cat(test_output, dim=0)


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

        rmse_error = mse(pred, target).sqrt_()
        self.rmse_meter.update(rmse_error)

    def metric_dict(self,):
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

    def metric_dict(self,):
        return {"Acc": self.acc_meter.avg,
                "F1": self.fscore_meter.avg}


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


def save_checkpoint(state, is_best, model_name, run_id):
    """
    Saves a checkpoint and overwrites the best model when is_best = True
    """
    checkpoint = f"models/{model_name}/checkpoint-r{run_id}.pth.tar"
    best = f"models/{model_name}/best-r{run_id}.pth.tar"

    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, best)


def load_previous_state(
    path, model, device, optimizer=None, normalizer=None, scheduler=None
):
    """
    """
    assert os.path.isfile(path), "no checkpoint found at '{}'".format(path)

    checkpoint = torch.load(path, map_location=device)
    start_epoch = checkpoint["epoch"]
    best_val_score = checkpoint["best_val_score"]
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if normalizer and model.task == "regression":
        normalizer.load_state_dict(checkpoint["normalizer"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    print("Loaded '{}'".format(path))

    return model, optimizer, normalizer, scheduler, start_epoch, best_val_score


def RobustL1(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = np.sqrt(2.0) * torch.abs(output - target) * torch.exp(-log_std) + log_std
    return torch.mean(loss)


def RobustL2(output, log_std, target):
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
    epsilon = torch.randn_like(sam_std)
    pre_logits = pre_logits.repeat_interleave(samples, dim=0) + \
                    torch.mul(epsilon, sam_std) 
    logits = softmax(pre_logits, dim=1).view(len(log_std), samples, -1)
    return torch.mean(logits, dim=1)