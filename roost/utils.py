import os
import gc
import torch
import shutil
import numpy as np
import torch.nn as nn

from tqdm.autonotebook import trange
from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse
from torch.nn.functional import mse_loss as mse


class BaseModelClass(nn.Module):
    """
    A base class for models.
    """
    def __init__(self, task, device=torch.device("cpu"), epoch=1, best_val_loss=None):
        super(BaseModelClass, self).__init__()
        self.task = task
        self.device = device
        self.epoch = epoch
        self.best_val_loss = best_val_loss

    def fit(self, train_generator, val_generator, optimizer,
            scheduler, epochs, criterion, normalizer,
            model_name, run_id, writer=None):
        start_epoch = self.epoch
        try:
            for epoch in range(start_epoch, start_epoch+epochs):
                self.epoch += 1
                # Training
                t_loss, t_metrics = self.evaluate(generator=train_generator,
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        normalizer=normalizer,
                                                        action="train",
                                                        verbose=True)

                print("Epoch: [{}/{}]".format(epoch, start_epoch + epochs - 1))
                print(f"Train      : Loss {t_loss:.4f}\t"+
                    "".join([f"{key} {val:.3f}\t" for key, val in t_metrics.items()]))

                # Validation
                if val_generator is not None:
                    with torch.no_grad():
                        # evaluate on validation set
                        v_loss, v_metrics = self.evaluate(generator=val_generator,
                                                          criterion=criterion,
                                                          optimizer=None,
                                                          normalizer=normalizer,
                                                          action="val")

                    print(f"Validation : Loss {v_loss:.4f}\t"+
                        "".join([f"{key} {val:.3f}\t" for key, val in v_metrics.items()]))

                    is_best = v_loss < self.best_val_loss
                    if is_best: self.best_val_loss = v_loss
                else:
                    is_best = False

                checkpoint_dict = {"state_dict": self.state_dict(),
                                    "epoch": self.epoch,
                                    "best_val_loss": self.best_val_loss,
                                    "optimizer": optimizer.state_dict(),
                                    "normalizer": normalizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    }

                save_checkpoint(checkpoint_dict,
                                is_best,
                                model_name,
                                run_id)

                if writer is not None:
                    writer.add_scalar("train/loss", t_loss, epoch+1)
                    for metric, val in t_metrics.items():
                        writer.add_scalar(f"train/{metric}", val, epoch+1)

                    if val_generator is not None:
                        writer.add_scalar("validation/loss", v_loss, epoch+1)
                        for metric, val in v_metrics.items():
                            writer.add_scalar(f"validation/{metric}", val, epoch+1)

                scheduler.step()

                # catch memory leak
                gc.collect()

        except KeyboardInterrupt:
            pass

        if writer is not None: 
            writer.close()

    def evaluate(self, generator, criterion, optimizer, normalizer, 
                    action="train", verbose=False):
        """
        evaluate the model
        """

        if action == "test":
            self.eval()
            test_targets = []
            test_pred = []
            test_std = []
            test_ids = []
            test_comp = []
        else:
            loss_meter = AverageMeter()
            rmse_meter = AverageMeter()
            mae_meter = AverageMeter()
            if action == "val":
                self.eval()
            elif action == "train":
                self.train()
            else:
                raise NameError("Only train, val or test is allowed as action")

        with trange(len(generator), disable=(not verbose)) as t:
            for input_, target, batch_comp, batch_ids in generator:

                # normalize target
                target_norm = normalizer.norm(target)

                # move tensors to GPU
                input_ = (tensor.to(self.device) for tensor in input_)
                target_norm = target_norm.to(self.device)

                # compute output
                output, log_std = self(*input_).chunk(2, dim=1)

                # get predictions and error
                pred = normalizer.denorm(output.data.cpu())

                if action == "test":
                    # get the aleatoric std
                    std = torch.exp(log_std).data.cpu()*normalizer.std

                    # collect the model outputs
                    test_ids += batch_ids
                    test_comp += batch_comp
                    test_targets += target.view(-1).tolist()
                    test_pred += pred.view(-1).tolist()
                    test_std += std.view(-1).tolist()

                else:
                    loss = criterion(output, log_std, target_norm)
                    loss_meter.update(loss.data.cpu().item(), target.size(0))

                    mae_error = mae(pred, target)
                    mae_meter.update(mae_error, target.size(0))

                    rmse_error = mse(pred, target).sqrt_()
                    rmse_meter.update(rmse_error, target.size(0))

                    if action == "train":
                        # compute gradient and do SGD step
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                t.update()

        if action == "test":
            return test_ids, test_comp, test_targets, test_pred, test_std
        else:
            return loss_meter.avg, {"MAE":mae_meter.avg, "RMSE":rmse_meter.avg}


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
        return {"mean": self.mean,
                "std": self.std}

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


def load_previous_state(path, model, device, optimizer=None,
                        normalizer=None, scheduler=None):
    """
    """
    assert os.path.isfile(path), "no checkpoint found at '{}'".format(path)

    checkpoint = torch.load(path, map_location=device)
    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"].cpu()
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if normalizer and model.task == "regression":
        normalizer.load_state_dict(checkpoint["normalizer"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    print("Loaded '{}'".format(path))

    return model, optimizer, normalizer, scheduler, start_epoch, best_val_loss


def RobustL1(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = np.sqrt(2.0) * torch.abs(output - target) * \
        torch.exp(- log_std) + log_std
    return torch.mean(loss)


def RobustL2(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = 0.5 * torch.pow(output - target, 2.0) * \
        torch.exp(- 2.0 * log_std) + log_std
    return torch.mean(loss)
