from __future__ import print_function
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.optim as optim
from future.backports import OrderedDict
import numpy as np
from models import wideresnet, resnet, resnet_imagenet, resnet_tiny200, mobilenet_v2, resnext_tiny200, resnext_imagenet, resnext_cifar, densenet_cifar, t2t_vit
from models import resnet_18_custom
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from utils import nn_util, tiny_datasets, imagenet_loader

from models.simple_vit import SimpleViT
from models.vit import ViT

parser = argparse.ArgumentParser(description='Training Neural Networks with Virtual Smoothing Classes')

parser.add_argument('--model_name', default='resnet-18',
                    help='Model name: mobilenet_v2, wrn-28-10, wrn-34-10, wrn-40-4, '
                         'resnet-18, resnet-50, resnext-50, resnext-101, resnext-29_2x64d, resnext-29_32x4d, densenet-121, '
                         't2t_vit_7, t2t_vit_10, t2t_vit_14, t2t_vit_19, t2t_vit_24')

parser.add_argument('--dataset', default='cifar10',
                    help='Dataset: svhn, cifar10, cifar100, tiny-imagenet-32x32, tiny-imagenet-64x64, imagenet, or mnist')

parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')

parser.add_argument('--optim', default='SGD',
                    help='Optimizer: SGD or AdamW')

parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W',
                    help='Weight decay coefficient (default: 1e-4)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='Number of epochs to train')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')

parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate is multiplied by gamma on schedule.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA training')
parser.add_argument('--model_dir', default='dnn_models/cifar/', help='Directory of model for saving checkpoints')
parser.add_argument('--gpuid', type=int, default=0, help='The ID of GPU.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='Number of batches to wait before logging training status')

parser.add_argument('--training_method', default='clean', help='Training method: clean or distill')
parser.add_argument('--resume_epoch', type=int, default=-1, metavar='N', help='Epoch for resuming training')

parser.add_argument('--alpha', default=0.7, type=float, help='Total confidence of virtual smoothing classes')
parser.add_argument('--v_classes', default=10, type=int,
                    help='The number of virtual smoothing classes')
parser.add_argument('--add_noise', action='store_true', default=False, help='Add noise to virtual classes')
parser.add_argument('--v_type', default='u', type=str, help='Type of noise added to virtual classes: u or n')
parser.add_argument('--vs_warmup', default=0, type=int, help='Warmup epoch for using VS labels.')

parser.add_argument('--teacher_model_name', default='wrn-34-10',
                    help='Teacher model name: wrn-28-10, wrn-34-10, wrn-40-4, resnet-18, or resnet-50')
parser.add_argument('--teacher_cpt_file', default='', help='File path of the teacher model')
parser.add_argument('--distill_type', default='real',
                    help='Learn knowledge from the real part or whole part of the teacher, options: real or whole')
parser.add_argument('--teacher_v_classes', default=10, type=int,
                    help='The number of virtual classes in the teacher model')
parser.add_argument('--d_alpha', default=1.0, type=float,
                    help='Parameter to balance the trade-off between the student model and the teacher model')
parser.add_argument('--temp', default=30.0, type=float, help='Temperature for distillation')

parser.add_argument('--use_mixup', action='store_true', default=False, help='Whether to use mixup')
parser.add_argument('--mixup_prob', default=-1, type=float, help="'Prob' from mixup_data_prob")
parser.add_argument('--use_randaug', action='store_true', default=False, help='Whether to use RandAugment')
parser.add_argument('--magnitude', type=int, default=9, metavar='N', help="'Magnitude' from v2.RandAugment")

parser.add_argument('--gpus', type=int, nargs='+', default=[], help='List of GPU IDs to use.')
parser.add_argument('--T_0', default=10, type=int, help='Parameter T_0 in CosineAnnealingWarmRestarts')
parser.add_argument('--T_mult', default=2, type=int, help='Parameter T_mult in CosineAnnealingWarmRestarts')
parser.add_argument('--training_logs', default="training_logs/cifar_10/resnet_18", type=str, help='Directory for training logs')
parser.add_argument('--base_width', default=64, type=int, help='Base width for ResNet-18 model')
parser.add_argument('--resnet_num_blocks', type=int, nargs='+', default=[2, 2, 2, 2], help='Number of blocks in each layer for ResNet-18 model')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
torch.cuda.set_device(int(args.gpuid))
if len(args.gpus) > 1:
    torch.cuda.set_device(int(args.gpus[0]))
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

if args.dataset == 'cifar10':
    NUM_REAL_CLASSES = 10
elif args.dataset == 'svhn':
    NUM_REAL_CLASSES = 10
elif args.dataset == 'cifar100':
    NUM_REAL_CLASSES = 100
elif 'tiny-imagenet' in args.dataset:
    NUM_REAL_CLASSES = 200
elif args.dataset == 'imagenet':
    NUM_REAL_CLASSES = 1000
elif args.dataset == 'mnist':
    # ------------------------------------------------------------------ #
    # MNIST: 10 digit classes (0-9)
    # ------------------------------------------------------------------ #
    NUM_REAL_CLASSES = 10
else:
    raise ValueError('error dataset: {0}'.format(args.dataset))

# settings
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.exists(args.training_logs):
    os.makedirs(args.training_logs)


def save_cpt(model, optimizer, epoch):
    path = os.path.join(args.model_dir, '{0}_model_epoch{1}.pt'.format(args.training_method, epoch))
    torch.save(model.state_dict(), path)
    path = os.path.join(args.model_dir, '{0}_cpt_epoch{1}.pt'.format(args.training_method, epoch))
    torch.save(optimizer.state_dict(), path)


def resume(model, optimizer, epoch):
    path = os.path.join(args.model_dir, '{0}_model_epoch{1}.pt'.format(args.training_method, epoch))
    model.load_state_dict(torch.load(path))
    path = os.path.join(args.model_dir, '{0}_cpt_epoch{1}.pt'.format(args.training_method, epoch))
    optimizer.load_state_dict(torch.load(path))
    return model, optimizer


def log_epoch(
    epoch, batch_time, epoch_lr, loss,
    train_nat_acc, train_nat_max_in_v, train_nat_max_in_v_corr,
    test_nat_acc, test_nat_max_in_v, test_nat_max_in_v_corr,
    msc, msc_v, std_corr_v_conf, mean_corr_v_conf
):
    if type(epoch_lr) == list:
        epoch_lr = epoch_lr[0]
    message = f"""
================================================================================
Epoch: {epoch}

| Metric                    | Value          |
|---------------------------|----------------|
| Time                      | {batch_time:.4f}
| Learning Rate             | {epoch_lr:.6f}
| Overall Train Loss        | {loss:.6f}
| Train Nat Acc             | {train_nat_acc:.4f}
| Train Max In V            | {train_nat_max_in_v:.4f}
| Train Max In V (Corr)     | {train_nat_max_in_v_corr:.4f}
| Test Nat Acc              | {test_nat_acc:.4f}
| Test Max In V             | {test_nat_max_in_v:.4f}
| Test Max In V (Corr)      | {test_nat_max_in_v_corr:.4f}
| MSC                       | {msc:.4f}
| MSC_V                     | {msc_v:.4f}
| Std Corr V Conf           | {std_corr_v_conf:.4f}
| Mean Corr V Conf          | {mean_corr_v_conf:.4f}
| alpha                     | {args.alpha}
================================================================================
"""
    with open(os.path.join(args.training_logs, 'log.txt'), 'a') as f:
        f.write(message)
    print(message)


def get_all_test_data(test_loader):
    x_test = None
    y_test = None
    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        if x_test is None:
            x_test = batch_x
        else:
            x_test = torch.cat((x_test, batch_x), 0)

        if y_test is None:
            y_test = batch_y
        else:
            y_test = torch.cat((y_test, batch_y), 0)

    return x_test, y_test


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    epoch_lr = args.lr
    for i in range(0, len(args.schedule)):
        if epoch > args.schedule[i]:
            epoch_lr = args.lr * np.power(args.gamma, (i + 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch_lr
    return epoch_lr


def add_noise_to_uniform(unif):
    assert len(unif.size()) == 2
    unif_elem = unif.float().mean()
    new_unif = unif.clone()
    new_unif.uniform_(unif_elem - 0.005 * unif_elem, unif_elem + 0.005 * unif_elem)
    factor = new_unif.sum(dim=1) / unif.sum(dim=1)
    new_unif = new_unif / factor.unsqueeze(dim=1)
    return new_unif


def label_smoothing(org_y, alpha, num_classes):
    y_ls = F.one_hot(org_y, num_classes=num_classes) * (1 - alpha) + (alpha / (num_classes))
    y_ls = y_ls.to(org_y.device)
    return y_ls


def constuct_vs_label(org_y, alpha, num_classes, v_classes, add_noise=False, v_type='u', sub_alpha=-1):
    assert len(org_y.size()) >= 1 and len(org_y.size()) <= 2
    if len(org_y.size()) == 1:
        if alpha == 0:
            return F.one_hot(org_y, num_classes=num_classes + v_classes).to(org_y.device)
        elif alpha > 0 and v_classes == 0:
            return label_smoothing(org_y, alpha, num_classes)
        elif alpha > 0 and v_classes > 0:
            y_vc = torch.zeros((len(org_y), num_classes + v_classes), device=org_y.device)
            u = [i for i in range(len(org_y))]
            y_vc[u, org_y] += (1 - alpha)
            if v_type == 'u':
                temp_v_conf = alpha / v_classes
                y_vc[:, num_classes:num_classes + v_classes] = temp_v_conf
                if add_noise:
                    y_vc[:, num_classes:num_classes + v_classes] = add_noise_to_uniform(
                        y_vc[:, num_classes:num_classes + v_classes])
            elif v_type == 'n':
                v_conf = torch.randint(low=0, high=100 * v_classes, size=(len(org_y), v_classes)).float().to(
                    org_y.device)
                v_conf = v_conf / v_conf.sum(dim=1).unsqueeze(dim=1)
                y_vc[:, num_classes:num_classes + v_classes] = alpha * v_conf
            return y_vc
        else:
            raise ValueError('alpha:{0} or v_classes:{1}'.format(alpha, v_classes))
    elif len(org_y.size()) == 2:
        y_vc = torch.zeros((len(org_y), num_classes + v_classes), device=org_y.device)
        if alpha == 0:
            y_vc[:, :num_classes] = org_y
            return y_vc
        elif alpha > 0 and v_classes == 0:
            y_vc = org_y * (1 - alpha)
            y_vc = y_vc + alpha / num_classes
            return y_vc
        elif alpha > 0 and v_classes > 0:
            y_vc[:, :num_classes] += org_y * (1 - alpha)
            y_vc[:, num_classes:] += alpha / v_classes
            return y_vc
        else:
            raise ValueError('alpha:{0}, v_classes:{1}'.format(alpha, v_classes))
    else:
        raise ValueError('org_y: {}'.format(org_y))


def train(model, train_loader, optimizer, scheduler, test_loader):
    start_epoch = 1
    if args.resume_epoch > 0:
        print('try to resume training from epoch', args.resume_epoch)
        model, optimizer = resume(model, optimizer, args.resume_epoch)
        start_epoch = args.resume_epoch + 1

    if args.teacher_cpt_file != '':
        teach_model = get_model(args.teacher_model_name, num_real_classes=NUM_REAL_CLASSES,
                                num_v_classes=args.teacher_v_classes, normalizer=None, dataset=args.dataset).to(device)
        teach_model.load_state_dict(torch.load(args.teacher_cpt_file))
        teach_model.eval()
        print('successfully loaded teach_model from {}'.format(args.teacher_cpt_file))
        KL_loss = nn.KLDivLoss(reduction='batchmean')

    iters = len(train_loader)
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        total = 0
        train_nat_correct = 0
        train_nat_max_in_v = 0
        train_nat_max_in_v_corr = 0
        total_loss = 0.0
        total_samples = 0
        for i, data in enumerate(train_loader):
            nat_batch_x, batch_y = data
            batch_y_hard = batch_y + 0
            if use_cuda:
                nat_batch_x = nat_batch_x.cuda(non_blocking=True)
                batch_y_hard = batch_y_hard.cuda(non_blocking=True)
                batch_y = batch_y.cuda(non_blocking=True)
            if args.use_mixup:
                if args.mixup_prob > 0:
                    nat_batch_x, batch_y = nn_util.mixup_data_prob(nat_batch_x, batch_y_hard, prob=args.mixup_prob,
                                                                    num_classes=NUM_REAL_CLASSES)
                else:
                    nat_batch_x, batch_y, lam = nn_util.mixup_data(nat_batch_x, batch_y_hard, alpha=0.8,
                                                                    num_classes=NUM_REAL_CLASSES)
            if epoch >= args.vs_warmup:
                batch_y_soft = constuct_vs_label(batch_y, args.alpha, NUM_REAL_CLASSES, args.v_classes, args.add_noise,
                                                 v_type=args.v_type)
            else:
                batch_y_soft = constuct_vs_label(batch_y, 0, NUM_REAL_CLASSES, args.v_classes, args.add_noise,
                                                 v_type=args.v_type)

            if args.teacher_cpt_file != '':
                with torch.no_grad():
                    nat_teach_logits = teach_model(nat_batch_x)

            if args.training_method == 'clean':
                model.train()
                nat_logits = model(nat_batch_x)
                loss = nn_util.cross_entropy_soft_target(nat_logits, batch_y_soft)
            elif args.training_method == 'distill':
                model.train()
                nat_stu_logits = model(nat_batch_x)
                if args.distill_type == 'real':
                    stu_teach_kl_loss = KL_loss(
                        F.log_softmax(nat_stu_logits[:, :NUM_REAL_CLASSES] / args.temp, dim=1),
                        F.softmax(nat_teach_logits[:, :NUM_REAL_CLASSES] / args.temp, dim=1))
                    nat_stu_ce_loss = nn_util.cross_entropy_soft_target(nat_stu_logits[:, :NUM_REAL_CLASSES],
                                                                        batch_y_soft[:, :NUM_REAL_CLASSES])
                else:
                    stu_teach_kl_loss = KL_loss(
                        F.log_softmax(nat_stu_logits / args.temp, dim=1),
                        F.softmax(nat_teach_logits / args.temp, dim=1))
                    nat_stu_ce_loss = nn_util.cross_entropy_soft_target(nat_stu_logits, batch_y_soft)
                loss = args.d_alpha * args.temp * args.temp * stu_teach_kl_loss + (1.0 - args.d_alpha) * (
                    nat_stu_ce_loss)
            else:
                raise ValueError('unsupported training method: {0}'.format(args.training_method))

            batch_size = len(batch_y_hard)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None and args.optim == 'AdamW_CAWR':
                scheduler.step(epoch + i / iters)
                epoch_lr = scheduler.get_lr()

            model.eval()
            with torch.no_grad():
                nat_logits = model(nat_batch_x)
                _, nat_pred = torch.max(nat_logits[:, :NUM_REAL_CLASSES], dim=1)
                nat_correct_indices = nat_pred == batch_y_hard
                train_nat_correct += nat_correct_indices.sum().item()
                _, nat_whole_preds = torch.max(nat_logits, dim=1)
                max_in_v_indices = nat_whole_preds >= NUM_REAL_CLASSES
                train_nat_max_in_v += max_in_v_indices.sum().item()
                train_nat_max_in_v_corr += (torch.logical_and(nat_correct_indices, max_in_v_indices)).sum().item()

            batch_size = len(batch_y_hard)
            total += batch_size
            if i % args.log_interval == 0 or i >= len(train_loader) - 1:
                processed_ratio = round((i / len(train_loader)) * 100, 2)
                print('Train Epoch: {}, Training progress: {}% [{}/{}], Loss: {}'.format(
                    epoch, processed_ratio, i, len(train_loader), loss))

        if args.optim == 'AdamW':
            scheduler.step()
            epoch_lr = scheduler.get_lr()
        elif args.optim == 'SGD':
            epoch_lr = adjust_learning_rate(optimizer, epoch)
        elif args.optim == 'AdamW_CAWR':
            pass
        else:
            raise ValueError('Error, unkonwn args.optim: {}'.format(args.optim))

        end_time = time.time()
        batch_time = end_time - start_time
        train_nat_acc = (float(train_nat_correct) / total) * 100

        test_nat_acc, test_nat_max_in_v, test_nat_max_in_v_corr, corr_conf, corr_v_conf = nn_util.eval(
            model, test_loader, NUM_REAL_CLASSES)
        msc = corr_conf.mean()
        msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
        if corr_v_conf.size(1) > 1:
            msc_v = corr_v_conf.max(dim=1)[0].mean()
            std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)

        epoch_loss = total_loss / total_samples
        log_epoch(epoch, batch_time, epoch_lr, epoch_loss, train_nat_acc, train_nat_max_in_v,
                  train_nat_max_in_v_corr, test_nat_acc, test_nat_max_in_v,
                  test_nat_max_in_v_corr, msc, msc_v, std_corr_v_conf, mean_corr_v_conf)
        save_cpt(model, optimizer, epoch)

    return loss.item(), test_nat_acc


# ============================================================================ #
#  Model factory — MNIST branch added                                          #
# ============================================================================ #
def get_model(model_name, num_real_classes, num_v_classes, normalizer=None,
              dataset='cifar10', base_width=64, resnet_num_blocks=[2, 2, 2, 2]):

    size_3x32x32  = ['svhn', 'cifar10', 'cifar100', 'tiny-imagenet-32x32']
    size_3x64x64  = ['tiny-imagenet-64x64']
    size_3x224x224 = ['imagenet']
    # ------------------------------------------------------------------ #
    # MNIST images are upsampled to 224×224 in the data loader below,    #
    # but they remain single-channel (1-channel / grayscale).            #
    # T2T-ViT accepts an `in_chans` kwarg which we set to 1 here.        #
    # ------------------------------------------------------------------ #
    size_1x224x224 = ['mnist']

    if dataset in size_3x32x32:
        if model_name == 'wrn-34-10':
            return wideresnet.WideResNet(depth=34, widen_factor=10, num_real_classes=num_real_classes,
                                         num_v_classes=num_v_classes, normalizer=normalizer)
        elif model_name == 'wrn-28-10':
            return wideresnet.WideResNet(depth=28, widen_factor=10, num_real_classes=num_real_classes,
                                         num_v_classes=num_v_classes, normalizer=normalizer)
        elif model_name == 'wrn-40-4':
            return wideresnet.WideResNet(depth=40, widen_factor=4, num_real_classes=num_real_classes,
                                         num_v_classes=num_v_classes, normalizer=normalizer)
        elif model_name == 'resnet-18':
            return resnet.ResNet18(num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                                   normalizer=normalizer)
        elif model_name == 'resnet-18-custom':
            return resnet_18_custom.ResNet18(num_blocks=resnet_num_blocks, base_width=base_width,
                                             num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                                             normalizer=normalizer)
        elif model_name == 'resnet-34':
            return resnet.ResNet34(num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                                   normalizer=normalizer)
        elif model_name == 'resnet-50':
            return resnet.ResNet50(num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                                   normalizer=normalizer)
        elif model_name == 'mobilenet_v2':
            return mobilenet_v2.mobilenet_v2(num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                                             normalizer=normalizer)
        elif model_name == 'resnext-50':
            return resnext_tiny200.resnext50_32x4d(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnext-29_2x64d':
            return resnext_cifar.ResNeXt29_2x64d(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnext-29_32x4d':
            return resnext_cifar.ResNeXt29_32x4d(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'densenet-121':
            return densenet_cifar.DenseNet121(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        else:
            raise ValueError('un-supported model: {0}', model_name)

    elif dataset in size_3x64x64:
        if model_name == 'resnet-18':
            return resnet_tiny200.resnet18(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-34':
            return resnet_tiny200.resnet34(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-50':
            return resnet_tiny200.resnet50(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnext-50':
            return resnext_tiny200.resnext50_32x4d(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        else:
            raise ValueError('un-supported model: {0}', model_name)

    elif dataset in size_3x224x224:
        if model_name == 'resnet-18':
            return resnet_imagenet.resnet18(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-34':
            return resnet_imagenet.resnet34(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-50':
            return resnet_imagenet.resnet50(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-101':
            return resnet_imagenet.resnet101(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnext-50':
            return resnext_imagenet.resnext50_32x4d(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnext-101':
            return resnext_imagenet.resnext101_32x4d(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'vit':
            model = ViT(image_size=224, patch_size=32, num_classes=num_real_classes + num_v_classes, dim=1024,
                        depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
            return model
        elif model_name == 'simple-vit':
            model = SimpleViT(image_size=224, patch_size=32, num_classes=num_real_classes + num_v_classes, dim=1024,
                              depth=6, heads=16, mlp_dim=2048)
            return model
        elif 't2t_' in model_name:
            model = _build_t2t_vit(model_name, num_real_classes, num_v_classes, in_chans=3)
            return model
        else:
            raise ValueError('un-supported model: {0}', model_name)

    # ------------------------------------------------------------------ #
    # NEW: MNIST branch — 1-channel 224×224 inputs                       #
    # Only T2T-ViT-14 and T2T-ViT-24 are required for Q2, but the full  #
    # T2T-ViT family is wired in for completeness.                       #
    # ------------------------------------------------------------------ #
    elif dataset in size_1x224x224:
        if 't2t_' in model_name:
            # Pass in_chans=1 so the first Unfold projection matches
            # the single grayscale channel of MNIST images.
            model = _build_t2t_vit(model_name, num_real_classes, num_v_classes, in_chans=1)
            return model
        else:
            raise ValueError(
                'Only T2T-ViT variants are supported for MNIST in this codebase. '
                'Unsupported model: {0}'.format(model_name)
            )

    else:
        raise ValueError('un-supported dataset: {0}', dataset)


# ============================================================================ #
#  Helper: build any T2T-ViT variant by name                                  #
# ============================================================================ #
def _build_t2t_vit(model_name, num_real_classes, num_v_classes, in_chans=3):
    """
    Instantiate a T2T-ViT model identified by `model_name`.

    The `in_chans` argument is forwarded to T2T_ViT so that single-channel
    inputs (e.g. MNIST) are handled correctly without any monkey-patching.
    The total number of output logits is num_real_classes + num_v_classes.
    """
    num_classes = num_real_classes + num_v_classes
    kwargs = dict(num_classes=num_classes, drop_path_rate=0.01, in_chans=in_chans)

    mapping = {
        't2t_vit_7':          (t2t_vit.t2t_vit_7,          {}),
        't2t_vit_10':         (t2t_vit.t2t_vit_10,         {}),
        't2t_vit_12':         (t2t_vit.t2t_vit_12,         {}),
        't2t_vit_14':         (t2t_vit.t2t_vit_14,         {}),
        't2t_vit_19':         (t2t_vit.t2t_vit_19,         {}),
        't2t_vit_24':         (t2t_vit.t2t_vit_24,         {}),
        't2t_vit_t_14':       (t2t_vit.t2t_vit_t_14,       {}),
        't2t_vit_t_19':       (t2t_vit.t2t_vit_t_19,       {}),
        't2t_vit_t_24':       (t2t_vit.t2t_vit_t_24,       {}),
        't2t_vit_14_resnext': (t2t_vit.t2t_vit_14_resnext, {}),
        't2t_vit_14_wide':    (t2t_vit.t2t_vit_14_wide,    {}),
    }

    if model_name not in mapping:
        raise ValueError('un-supported T2T-ViT variant: {0}'.format(model_name))

    factory_fn, extra_kwargs = mapping[model_name]
    return factory_fn(**{**kwargs, **extra_kwargs})


def main():
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    if args.dataset == 'cifar10':
        transform_train = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        transform_test = T.Compose([T.ToTensor()])
        dataloader = torchvision.datasets.CIFAR10
        train_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar10/', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar10/', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'cifar100':
        transform_train = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        transform_test = T.Compose([T.ToTensor()])
        dataloader = torchvision.datasets.CIFAR100
        train_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar100/', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar100/', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'svhn':
        transform_train = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        svhn_train = torchvision.datasets.SVHN(root='../../datasets/svhn/', download=True,
                                               transform=transform_train, split='train')
        svhn_test  = torchvision.datasets.SVHN(root='../../datasets/svhn/', download=True,
                                               transform=transform_test,  split='test')
        train_loader = torch.utils.data.DataLoader(svhn_train, batch_size=args.batch_size, shuffle=True,  **kwargs)
        test_loader  = torch.utils.data.DataLoader(svhn_test,  batch_size=args.batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'tiny-imagenet-32x32':
        data_dir  = '../../datasets/tiny-imagenet-200/'
        normalize = T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform_train = T.Compose([
            T.RandomResizedCrop(32),
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor(), normalize
        ])
        transform_test = T.Compose([T.Resize(32), T.ToTensor(), normalize])
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        testset  = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'),   transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,  **kwargs)
        test_loader  = torch.utils.data.DataLoader(testset,  batch_size=args.batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'tiny-imagenet-64x64':
        data_dir  = '../../datasets/tiny-imagenet-200/'
        normalize = T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform_train = T.Compose([
            T.Pad(8, padding_mode='reflect'),
            T.RandomCrop(64),
            T.RandomHorizontalFlip(),
            T.ToTensor(), normalize
        ])
        transform_test = T.Compose([T.ToTensor(), normalize])
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        testset  = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'),   transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,  **kwargs)
        test_loader  = torch.utils.data.DataLoader(testset,  batch_size=args.batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'imagenet':
        imagenet_root = '../../datasets/'
        train_loader, test_loader = imagenet_loader.data_loader(
            imagenet_root, batch_size=args.batch_size, workers=16,
            randaug=args.use_randaug, magnitude=args.magnitude)

    # ------------------------------------------------------------------ #
    # MNIST data pipeline                                                 #
    #                                                                     #
    # Design decisions:                                                   #
    #  1. Resize 28×28 → 224×224 so T2T-ViT's patch-embedding works      #
    #     without any architectural change (T2T_module stride = 4×2×2    #
    #     = 16, giving 14×14 = 196 tokens, matching the default config). #
    #  2. Keep images single-channel; the model is built with in_chans=1.#
    #  3. Normalize with the empirical MNIST mean/std (0.1307 / 0.3081). #
    #  4. Apply mild augmentation (random crop + horizontal flip) to      #
    #     reduce over-fitting on the small 28×28 source images.          #
    # ------------------------------------------------------------------ #
    elif args.dataset == 'mnist':
        # mnist_mean = (0.1307,)          # per-channel mean for grayscale
        # mnist_std  = (0.3081,)          # per-channel std  for grayscale

        # transform_train = T.Compose([
        #     # Upsample to the spatial resolution T2T-ViT was designed for.
        #     T.Resize(224),
        #     # Mild spatial jitter to improve generalisation.
        #     T.RandomCrop(224, padding=8),
        #     # MNIST digits are symmetric; horizontal flip is valid.
        #     T.RandomHorizontalFlip(),
        #     T.ToTensor(),
        #     T.Normalize(mnist_mean, mnist_std),
        # ])

        # transform_test = T.Compose([
        #     T.Resize(224),
        #     T.ToTensor(),
        #     T.Normalize(mnist_mean, mnist_std),
        # ])
        transform_train = T.Compose([
            T.Resize(224),
            T.Grayscale(num_output_channels=3),
            T.RandomCrop(224, padding=4),
            T.ToTensor(),
            T.Normalize((0.1307,)*3, (0.3081,)*3),
        ])

        transform_test = T.Compose([
            T.Resize(224),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize((0.1307,)*3, (0.3081,)*3),
        ])

        mnist_train = torchvision.datasets.MNIST(
            root='../../datasets/mnist/', train=True,  download=True, transform=transform_train)
        mnist_test  = torchvision.datasets.MNIST(
            root='../../datasets/mnist/', train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(
            mnist_train, batch_size=args.batch_size, shuffle=True,  **kwargs)
        test_loader  = torch.utils.data.DataLoader(
            mnist_test,  batch_size=args.batch_size, shuffle=False, **kwargs)

    else:
        raise ValueError('un-supported dataset: {0}'.format(args.dataset))

    cudnn.benchmark = True

    model = get_model(args.model_name, num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes,
                      normalizer=None, dataset=args.dataset,
                      base_width=args.base_width, resnet_num_blocks=args.resnet_num_blocks)

    if len(args.gpus) > 1:
        model = nn.DataParallel(model.to(device), device_ids=args.gpus, output_device=args.gpus[0])
    else:
        model = model.to(device)

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        scheduler = None
    elif args.optim == 'AdamW':
        optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': 1e-3}],
                                lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5,
                                      last_epoch=args.resume_epoch)
    elif args.optim == 'AdamW_CAWR':
        optimizer = optim.AdamW([{'params': model.parameters(), 'initial_lr': 1e-3}],
                                lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult,
                                               eta_min=1e-5, last_epoch=args.resume_epoch)
    elif args.optim == 'Adam':
        optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': 1e-3}],
                               lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5,
                                      last_epoch=args.resume_epoch)
    else:
        raise ValueError('Error, un-supported optimizer: {}'.format(args.optim))

    print('=' * 88)
    print('args:', args)
    print('=' * 88)

    train(model, train_loader, optimizer, scheduler, test_loader)


if __name__ == '__main__':
    main()