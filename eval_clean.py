from __future__ import print_function
import os
import torch
import argparse
import numpy as np

import torchvision
import torchvision.transforms as T
from torchvision import datasets
import torch.nn.functional as F
from models import wideresnet, resnet, resnet_imagenet, resnet_tiny200, mobilenet_v2, resnext_tiny200, resnext_imagenet, resnext_cifar, densenet_cifar, t2t_vit
from models import resnet_18_custom
from utils import nn_util, tiny_datasets, imagenet_loader
from models.simple_vit import SimpleViT
from models.vit import ViT

parser = argparse.ArgumentParser(description='PyTorch Evaluation')
parser.add_argument('--model_name', default='resnet-18',
                    help='Model name: mobilenet_v2, wrn-28-10, wrn-34-10, wrn-40-4, '
                         'resnet-18, resnet-50, resnext-50, resnext-101, resnext-29_2x64d, resnext-29_32x4d, densenet-121, '
                         't2t_vit_7, t2t_vit_10, t2t_vit_14, t2t_vit_19')
parser.add_argument('--dataset', default='cifar10',
                    help='Dataset: svhn, cifar10, cifar100, tiny-imagenet-32x32, tiny-imagenet-64x64, or imagenet')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA training')
parser.add_argument('--gpuid', type=int, default=0, help='The ID of GPU.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument('--model_file', default='./dnn_models/dataset/model.pt', help='file path of the model')
parser.add_argument('--v_classes', default=10, type=int,
                    help='The number of virtual smoothing classes')
parser.add_argument('--alpha', default=0., type=float, help='Total confidence of virtual smoothing classes')
parser.add_argument('--temp', default=1.0, type=float, help='temperature scaling')
parser.add_argument('--base_width', default=64, type=int, help='Base width for ResNet-18 model')
parser.add_argument('--resnet_num_blocks', type=int, nargs='+', default=[2,2,2,2], help='Number of blocks in each layer for ResNet-18 model')
parser.add_argument('--testing_logs', default="testing_logs/cifar_10/resnet_18", type=str, help='Directory for testing logs')
parser.add_argument('--final_epoch', default=200, type=int, help='Final epoch number for log naming')
args = parser.parse_args()

if args.dataset == 'cifar10':
    NUM_REAL_CLASSES = 10
    NUM_EXAMPLES = 50000
elif args.dataset == 'svhn':
    NUM_REAL_CLASSES = 10
    NUM_EXAMPLES = 73257
elif args.dataset == 'cifar100':
    NUM_REAL_CLASSES = 100
    NUM_EXAMPLES = 50000
elif 'tiny-imagenet' in args.dataset:
    NUM_REAL_CLASSES = 200
    # NUM_EXAMPLES = 50000
elif args.dataset == 'imagenet':
    NUM_REAL_CLASSES = 1000
    # NUM_EXAMPLES = 50000
elif args.dataset == 'mnist':
    NUM_REAL_CLASSES = 10
else:
    raise ValueError('error dataset: {0}'.format(args.dataset))

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
storage_device = torch.device("cpu")
normalizer = None
if not os.path.exists(args.testing_logs):
    os.makedirs(args.testing_logs)

def get_model(model_name, num_real_classes, num_v_classes, normalizer=None, dataset='cifar10'):
    size_3x32x32 = ['svhn', 'cifar10', 'cifar100', 'tiny-imagenet-32x32']
    size_3x64x64 = ['tiny-imagenet-64x64']
    size_3x224x224 = ['imagenet']
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
            return resnet_18_custom.ResNet18(num_blocks=args.resnet_num_blocks, base_width=args.base_width, num_real_classes=num_real_classes, num_v_classes=num_v_classes,
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
            model = ViT(image_size=224, patch_size=32, num_classes=num_real_classes + num_v_classes, dim=1024, depth=6,
                        heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1)
            return model
        elif model_name == 'simple-vit':
            model = SimpleViT(image_size=224, patch_size=32, num_classes=num_real_classes + num_v_classes, dim=1024,
                              depth=6, heads=16, mlp_dim=2048)
            return model
        elif 't2t_' in model_name:
            if model_name == 't2t_vit_7':
                model = t2t_vit.t2t_vit_7(num_classes=num_real_classes + num_v_classes, drop_path_rate=0.01)
            elif model_name == 't2t_vit_10':
                model = t2t_vit.t2t_vit_10(num_classes=num_real_classes + num_v_classes, drop_path_rate=0.01)
            elif model_name == 't2t_vit_12':
                model = t2t_vit.t2t_vit_12(num_classes=num_real_classes + num_v_classes, drop_path_rate=0.01)
            elif model_name == 't2t_vit_14':
                model = t2t_vit.t2t_vit_14(num_classes=num_real_classes + num_v_classes, drop_path_rate=0.01)
            elif model_name == 't2t_vit_19':
                model = t2t_vit.t2t_vit_19(num_classes=num_real_classes + num_v_classes, drop_path_rate=0.01)
            elif model_name == 't2t_vit_24':
                model = t2t_vit.t2t_vit_24(num_classes=num_real_classes + num_v_classes, drop_path_rate=0.01)
            elif model_name == 't2t_vit_t_14':
                model = t2t_vit.t2t_vit_t_14(num_classes=num_real_classes + num_v_classes, drop_path_rate=0.01)
            elif model_name == 't2t_vit_t_19':
                model = t2t_vit.t2t_vit_t_19(num_classes=num_real_classes + num_v_classes, drop_path_rate=0.01)
            elif model_name == 't2t_vit_t_24':
                model = t2t_vit.t2t_vit_t_24(num_classes=num_real_classes + num_v_classes, drop_path_rate=0.01)
            elif model_name == 't2t_vit_14_resnext':
                model = t2t_vit.t2t_vit_14_resnext(num_classes=num_real_classes + num_v_classes, drop_path_rate=0.01)
            elif model_name == 't2t_vit_14_wide':
                model = t2t_vit.t2t_vit_14_wide(num_classes=num_real_classes + num_v_classes, drop_path_rate=0.01)
            # return get_t2t_vit(args)
            return model
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
        raise ValueError('un-supported dataset: {0}'.format(dataset))


def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


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

def print_ece_results(ece, bin_info, logs_dir=None):
    
    lines = []
    lines.append("\n" + "="*75)
    lines.append(f"  Dataset: {args.dataset}")
    lines.append(f"  Model: {args.model_name}")
    lines.append(f"  Virtual Smoothing Classes: {args.v_classes}")
    lines.append(f"  Alpha (Total Confidence of Virtual Classes): {args.alpha}")
    lines.append(f"  Max Epoch Trained: {args.final_epoch}")
    if args.model_name == 'resnet-18-custom':
        
        lines.append(f"  ResNet-18 Custom Configuration:")
        lines.append(f"      num_blocks: {args.resnet_num_blocks}")
        lines.append(f"      base_width: {args.base_width}")
    lines.append(f"  Expected Calibration Error (ECE): {ece.item():.4f}")

    bin_width = bin_info['bin_width']
    num_bins = bin_info['num']

    total_samples = 0
    correct_samples = 0
    for i in range(num_bins):
        bin_lower = round(i * bin_width, 10)
        if bin_lower not in bin_info:
            continue
        b = bin_info[bin_lower]
        num = b['num']
        total_samples += num
        if num > 0:
            correct_samples += b['acc'] * num

    overall_accuracy = correct_samples / total_samples if total_samples > 0 else 0
    lines.append(f"  Overall Accuracy:                 {overall_accuracy:.4f} ({overall_accuracy:.1%})")
    lines.append("="*75)
    lines.append(f"  {'Bin Range':<18} {'Samples':>8} {'Accuracy':>10} {'Avg Conf':>10} {'Gap':>8}")
    lines.append("-"*75)

    for i in range(num_bins):
        bin_lower = round(i * bin_width, 10)
        bin_upper = round(bin_lower + bin_width, 10)
        if bin_lower not in bin_info:
            continue
        b = bin_info[bin_lower]
        num = b['num']
        label = f"  {bin_lower:.1f} – {bin_upper:.1f}"
        if num == 0:
            lines.append(f"{label:<18} {'0':>8} {'—':>10} {'—':>10} {'—':>8}")
        else:
            acc = b['acc']
            conf = b['avg_conf']
            gap = acc - conf
            bar = "█" * int(acc * 20)
            lines.append(f"{label:<18} {num:>8} {acc:>9.1%} {conf:>9.1%} {gap:>+8.3f}  {bar}")

    lines.append("-"*75)
    lines.append(f"  {'Total Samples':<17} {total_samples:>8}")
    lines.append("="*75 + "\n")

    # --- Print to terminal ---
    for line in lines:
        print(line)

    # --- Save to log file ---
    # --- Save to log file ---
    if logs_dir:
        # os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, "log.txt")
        with open(log_path, 'w',encoding='utf-8') as f:
            for line in lines:
                f.write(line + "\n")
        print(f"  [LOG] Saved to {log_path}")

    
def cal_ece(model, test_loader):
    y_test = torch.tensor([]).to(device)
    probs = torch.tensor([]).to(device)
    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        with torch.no_grad():
            batch_logits = model(batch_x)
            # batch_probs = F.softmax(batch_logits, dim=1)
            batch_probs = F.softmax(batch_logits[:, :NUM_REAL_CLASSES]/args.temp, dim=1)
            if args.alpha > 0 and args.v_classes == 0:
                mins = batch_probs.min(dim=1)[0]
                for i in range(len(batch_probs)):
                    batch_probs[i] = batch_probs[i] - mins[i]
                    batch_probs[i] = batch_probs[i] / batch_probs[i].sum()
        probs = torch.cat((probs, batch_probs), dim=0)
        y_test = torch.cat((y_test, batch_y), dim=0)
    ece, bin_info = expected_calibration_error(probs.cpu().numpy(), y_test.cpu().numpy(), M=10)
    # print(ece, bin_info)
    print_ece_results(ece, bin_info,args.testing_logs)


def expected_calibration_error(samples, true_labels, M=5):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # get max probability per sample i
    confidences = np.max(samples[:, :NUM_REAL_CLASSES], axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples[:, :NUM_REAL_CLASSES], axis=1)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels
    bin_info = {}
    bin_info['num'] = M
    bin_info['bin_width'] = bin_boundaries[1] - bin_boundaries[0]
    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        cur_bin_infp = {}
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()
        cur_bin_infp['num'] = in_bin.sum()
        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
            cur_bin_infp['acc'] = accuracy_in_bin
            cur_bin_infp['avg_conf'] = avg_confidence_in_bin
            bin_info[bin_lower] = cur_bin_infp
        else:
            bin_info[bin_lower] = cur_bin_infp
    return ece, bin_info

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
    # setup data loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        transform_test = T.Compose([T.ToTensor()])
        test_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar10', train=False, download=True, transform=transform_test), batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        transform_test = T.Compose([T.ToTensor()])
        test_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar100', train=False, download=True, transform=transform_test), batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'svhn':
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        svhn_test = torchvision.datasets.SVHN(root='../../datasets/svhn', download=True, transform=transform_test, split='test')
        test_loader = torch.utils.data.DataLoader(dataset=svhn_test, batch_size=args.batch_size, shuffle=False,
                                                  **kwargs)
    elif args.dataset == 'tiny-imagenet-64x64':
        data_dir = '../../datasets/tiny-imagenet-200/'
        normalize = T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform_test = T.Compose([
            T.ToTensor(),
            normalize
        ])
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'imagenet':
        imagenet_root = '../../datasets/'
        train_loader, test_loader = imagenet_loader.data_loader(imagenet_root, batch_size=args.batch_size)
    elif args.dataset == "mnist":
        transform_test = T.Compose([
            T.Resize(224),
            
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
        ])
        mnist_test  = torchvision.datasets.MNIST(
            root='../../datasets/mnist/', train=False, download=True, transform=transform_test)
        test_loader  = torch.utils.data.DataLoader(
            mnist_test,  batch_size=args.batch_size, shuffle=False, **kwargs)
    
    else:
        raise ValueError('error dataset: {0}'.format(args.dataset))

    print('================================================================')
    print('args:', args)
    print('================================================================')
    model = get_model(args.model_name, num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes, normalizer=normalizer, dataset=args.dataset)
    cpt = filter_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))
    model.load_state_dict(cpt)
    model = model.to(device)
    cal_ece(model, test_loader)

if __name__ == '__main__':
    main()

