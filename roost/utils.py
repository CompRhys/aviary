import os
import gc
import torch
import shutil
import numpy as np
import torch.nn as nn

from tqdm.autonotebook import trange
from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse


class BaseModelClass(nn.Module):
    """
    A base class for models.
    """

    def __init__(self, device=torch.device("cpu"), epoch=1, best_mae=None):
        super(BaseModelClass, self).__init__()
        self.device = device
        self.best_mae = best_mae  # large number as placeholder
        self.epoch = epoch
        pass

    def fit(self, train_generator, val_generator, optimizer,
            scheduler, epochs, criterion, normalizer, writer,
            checkpoint_file, best_file):
        start_epoch = self.epoch
        try:
            for epoch in range(start_epoch, start_epoch+epochs):
                self.epoch += 1
                # Training
                t_loss, t_mae, t_rmse = self.evaluate(generator=train_generator,
                                                        criterion=criterion,
                                                        optimizer=optimizer,
                                                        normalizer=normalizer,
                                                        task="train",
                                                        verbose=True)

                # Validation
                with torch.no_grad():
                    # evaluate on validation set
                    v_loss, v_mae, v_rmse = self.evaluate(generator=val_generator,
                                                          criterion=criterion,
                                                          optimizer=None,
                                                          normalizer=normalizer,
                                                          task="val")

                # if epoch % args.print_freq == 0:
                print("Epoch: [{}/{}]\n"
                        "Train      : Loss {:.4f}\t"
                        "MAE {:.3f}\t RMSE {:.3f}\n"
                        "Validation : Loss {:.4f}\t"
                        "MAE {:.3f}\t RMSE {:.3f}\n".format(
                        epoch, start_epoch + epochs - 1,
                        t_loss, t_mae, t_rmse,
                        v_loss, v_mae, v_rmse))

                is_best = v_mae < self.best_mae
                if is_best:
                    self.best_mae = v_mae

                checkpoint_dict = {"state_dict": self.state_dict(),
                                    "epoch": self.epoch,
                                    "best_error": self.best_mae,
                                    "optimizer": optimizer.state_dict(),
                                    "normalizer": normalizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    }

                save_checkpoint(checkpoint_dict,
                                is_best,
                                checkpoint_file,
                                best_file)

                writer.add_scalar("loss/train", t_loss, epoch+1)
                writer.add_scalar("loss/validation", v_loss, epoch+1)
                writer.add_scalar("rmse/train", t_rmse, epoch+1)
                writer.add_scalar("rmse/validation", v_rmse, epoch+1)
                writer.add_scalar("mae/train", t_mae, epoch+1)
                writer.add_scalar("mae/validation", v_mae, epoch+1)

                scheduler.step()

                # catch memory leak
                gc.collect()

        except KeyboardInterrupt:
            pass

        writer.close()

    def evaluate(self, generator, criterion, optimizer,
                    normalizer, task="train", verbose=False):
        """
        evaluate the model
        """

        if task == "test":
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
            if task == "val":
                self.eval()
            elif task == "train":
                self.train()
            else:
                raise NameError("Only train, val or test is allowed as task")

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

                if task == "test":
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

                    if task == "train":
                        # compute gradient and do SGD step
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                t.update()

        if task == "test":
            return test_ids, test_comp, test_targets, test_pred, test_std
        else:
            return loss_meter.avg, mae_meter.avg, rmse_meter.avg

    def predict(self, generator, criterion, optimizer,
                normalizer, verbose=False):
        """
        alias to evaluate on test set
        """
        return self.evaluate(generator, criterion, optimizer,
                                normalizer, verbose, task="test")


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


def save_checkpoint(state, is_best,
                    checkpoint="checkpoint.pth.tar",
                    best="best.pth.tar"):
    """
    Saves a checkpoint and overwrites the best model when is_best = True
    """

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
    best_error = checkpoint["best_error"].cpu()
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if normalizer:
        normalizer.load_state_dict(checkpoint["normalizer"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    print("Loaded '{}'".format(path))

    return model, optimizer, normalizer, scheduler, start_epoch, best_error


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
