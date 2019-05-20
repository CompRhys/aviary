import torch
from tqdm import trange
import shutil
import csv

from torch.nn.functional import l1_loss as mae
from torch.nn.functional import mse_loss as mse
from sampnn.data import AverageMeter, Normalizer


def train(train_loader, model, criterion, optimizer, 
          normalizer, verbose = False, cuda=False):
    """
    run a forward pass, backwards pass and then update weights
    """
    losses = AverageMeter()
    mae_errors = AverageMeter()

    with trange(len(train_loader)) as t:
        for i, (input_, target, _) in enumerate(train_loader):
            
            # normalize target
            target_var = normalizer.norm(target)
            
            if cuda:
                input_ = (input_[0].cuda(async=True),
                            input_[1].cuda(async=True),
                            input_[2].cuda(async=True),
                            input_[3].cuda(async=True),
                            input_[4].cuda(async=True),
                            input_[5].cuda(async=True))
                target_var = target_var.cuda(async=True)

            # compute output
            output = model(*input_)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            mse_error = torch.sqrt(mse(normalizer.denorm(output.data.cpu()), target))
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t.set_postfix(loss=losses.avg)
            t.update()

    return losses.avg, mae_errors.avg
    

def evaluate(generator, model, criterion, normalizer, 
                test=False, verbose=False, cuda=False):
    """ evaluate the model """
    losses = AverageMeter()
    errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    if test:
       label = 'Test'
    else:
        label = 'Validate'

    for i, (input_, target, batch_cif_ids) in enumerate(generator):
        
        # normalize target
        target_var = normalizer.norm(target)
        
        if cuda:
            input_ = (input_[0].cuda(async=True),
                        input_[1].cuda(async=True),
                        input_[2].cuda(async=True),
                        input_[3].cuda(async=True),
                        input_[4].cuda(async=True),
                        input_[5].cuda(async=True))
            target_var = target_var.cuda(async=True)

        # compute output
        output = model(*input_)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        mse_error = mse(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        errors.update(mse_error, target.size(0))

        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

    if test:  
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))

        print('Test : Loss {loss.avg:.4f}\t'
                    'MAE {error.avg:.3f}\n'.format(loss=losses, error=errors))

        pass
    else:
        return losses.avg, errors.avg
                                                

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    '''
    Saves a checkpoint and overwrites the best model when is_best = True
    '''
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')