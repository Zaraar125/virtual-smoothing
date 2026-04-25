"""
Q5 Experiment: Effect of Including Virtual Classes in Final Prediction Space

This script compares two prediction strategies at test time:
  (A) CORRECT:   Softmax over real classes only  → argmax over [:NUM_REAL_CLASSES]
  (B) INCORRECT: Softmax over all classes        → argmax over [:NUM_REAL_CLASSES + v_classes]

The incorrect approach allows virtual class logits to distort the probability
distribution, potentially "stealing" confidence from real classes and
degrading accuracy and calibration.
"""

from __future__ import print_function
import os
import torch
import argparse
import numpy as np

import torchvision
import torchvision.transforms as T
from torchvision import datasets
import torch.nn.functional as F

# ── Re-use model/utility imports from the original codebase ──────────────────
from models import wideresnet, resnet, resnet_imagenet, resnet_tiny200
from models import mobilenet_v2, resnext_tiny200, resnext_cifar
from models import densenet_cifar, t2t_vit
from models import resnet_18_custom
from utils import nn_util, imagenet_loader

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Q5: Correct vs Incorrect Use of Virtual Classes at Test Time')

parser.add_argument('--model_name', default='resnet-18',
                    help='Model architecture (same options as eval.py)')
parser.add_argument('--dataset', default='cifar10',
                    help='Dataset: cifar10, cifar100, svhn, tiny-imagenet-64x64, imagenet')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpuid', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model_file', default='./dnn_models/dataset/model.pt',
                    help='Path to trained model checkpoint')
parser.add_argument('--v_classes', default=10, type=int,
                    help='Number of virtual smoothing classes used during training')
parser.add_argument('--alpha', default=0.1, type=float,
                    help='Total confidence allocated to virtual classes during training')
parser.add_argument('--temp', default=1.0, type=float,
                    help='Temperature for scaling (applied to real-class logits)')
parser.add_argument('--base_width', default=64, type=int)
parser.add_argument('--resnet_num_blocks', type=int, nargs='+', default=[2, 2, 2, 2])
parser.add_argument('--testing_logs', default='testing_logs/q5_experiment', type=str)
parser.add_argument('--final_epoch', default=200, type=int)

args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
DATASET_CONFIG = {
    'cifar10':  {'num_real': 10,   'num_examples': 10000},
    'svhn':     {'num_real': 10,   'num_examples': 26032},
    'cifar100': {'num_real': 100,  'num_examples': 10000},
    'imagenet': {'num_real': 1000, 'num_examples': 50000},
}
for k in list(DATASET_CONFIG):
    if 'tiny-imagenet' in k:
        DATASET_CONFIG[k] = {'num_real': 200, 'num_examples': 10000}

if args.dataset in DATASET_CONFIG:
    NUM_REAL_CLASSES = DATASET_CONFIG[args.dataset]['num_real']
elif 'tiny-imagenet' in args.dataset:
    NUM_REAL_CLASSES = 200
else:
    raise ValueError(f'Unsupported dataset: {args.dataset}')

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

os.makedirs(args.testing_logs, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model loader  (copied & trimmed from eval.py)
# ─────────────────────────────────────────────────────────────────────────────
def get_model(model_name, num_real_classes, num_v_classes, normalizer=None, dataset='cifar10'):
    size_3x32x32 = ['svhn', 'cifar10', 'cifar100', 'tiny-imagenet-32x32']
    size_3x64x64 = ['tiny-imagenet-64x64']
    size_3x224x224 = ['imagenet']

    if dataset in size_3x32x32:
        if model_name == 'wrn-34-10':
            return wideresnet.WideResNet(34, 10, num_real_classes, num_v_classes, normalizer)
        elif model_name == 'wrn-28-10':
            return wideresnet.WideResNet(28, 10, num_real_classes, num_v_classes, normalizer)
        elif model_name == 'wrn-40-4':
            return wideresnet.WideResNet(40, 4, num_real_classes, num_v_classes, normalizer)
        elif model_name == 'resnet-18':
            return resnet.ResNet18(num_real_classes=num_real_classes,
                                   num_v_classes=num_v_classes, normalizer=normalizer)
        elif model_name == 'resnet-18-custom':
            return resnet_18_custom.ResNet18(num_blocks=args.resnet_num_blocks,
                                             base_width=args.base_width,
                                             num_real_classes=num_real_classes,
                                             num_v_classes=num_v_classes,
                                             normalizer=normalizer)
        elif model_name == 'resnet-34':
            return resnet.ResNet34(num_real_classes=num_real_classes,
                                   num_v_classes=num_v_classes, normalizer=normalizer)
        elif model_name == 'resnet-50':
            return resnet.ResNet50(num_real_classes=num_real_classes,
                                   num_v_classes=num_v_classes, normalizer=normalizer)
        elif model_name == 'mobilenet_v2':
            return mobilenet_v2.mobilenet_v2(num_real_classes=num_real_classes,
                                             num_v_classes=num_v_classes, normalizer=normalizer)
        elif model_name == 'resnext-29_2x64d':
            return resnext_cifar.ResNeXt29_2x64d(num_real_classes, num_v_classes)
        elif model_name == 'resnext-29_32x4d':
            return resnext_cifar.ResNeXt29_32x4d(num_real_classes, num_v_classes)
        elif model_name == 'densenet-121':
            return densenet_cifar.DenseNet121(num_real_classes, num_v_classes)
        else:
            raise ValueError(f'Unsupported model: {model_name}')
    elif dataset in size_3x64x64:
        if model_name == 'resnet-18':
            return resnet_tiny200.resnet18(num_real_classes, num_v_classes)
        elif model_name == 'resnet-50':
            return resnet_tiny200.resnet50(num_real_classes, num_v_classes)
        else:
            raise ValueError(f'Unsupported model: {model_name}')
    elif dataset in size_3x224x224:
        if model_name == 'resnet-18':
            return resnet_imagenet.resnet18(num_real_classes, num_v_classes)
        elif model_name == 'resnet-50':
            return resnet_imagenet.resnet50(num_real_classes, num_v_classes)
        else:
            raise ValueError(f'Unsupported model: {model_name}')
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')


def filter_state_dict(state_dict):
    from collections import OrderedDict
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        new_state_dict[k[7:] if 'module' in k else k] = v
    return new_state_dict

# ─────────────────────────────────────────────────────────────────────────────
# ECE helper
# ─────────────────────────────────────────────────────────────────────────────
def expected_calibration_error(confidences, predicted_labels, true_labels, M=10):
    """Generic ECE given pre-computed confidences and predictions."""
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    accuracies = (predicted_labels == true_labels)

    bin_info = {'num': M, 'bin_width': bin_boundaries[1] - bin_boundaries[0]}
    ece = 0.0
    for bl, bu in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bl, confidences <= bu)
        prob_in_bin = in_bin.mean()
        bin_info[round(bl, 10)] = {'num': int(in_bin.sum())}
        if prob_in_bin > 0:
            acc_in_bin  = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += abs(acc_in_bin - conf_in_bin) * prob_in_bin
            bin_info[round(bl, 10)].update({'acc': acc_in_bin, 'avg_conf': conf_in_bin})
    return ece, bin_info

# ─────────────────────────────────────────────────────────────────────────────
# Core experiment
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(model, test_loader):
    """
    Collect raw logits for the entire test set, then evaluate under two
    prediction strategies and compare results side-by-side.
    """
    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits  = model(batch_x)          # shape: (B, NUM_REAL + v_classes)
            all_logits.append(logits.cpu())
            all_labels.append(batch_y)

    all_logits = torch.cat(all_logits, dim=0)  # (N, C_total)
    all_labels = torch.cat(all_labels, dim=0).numpy()

    C_real  = NUM_REAL_CLASSES
    C_total = C_real + args.v_classes

    # ── Strategy A: CORRECT ──────────────────────────────────────────────────
    # Slice out only the real-class logits, apply temperature, then softmax.
    # Virtual class scores are completely ignored.
    real_logits   = all_logits[:, :C_real] / args.temp        # (N, C_real)
    probs_correct = F.softmax(real_logits, dim=1).numpy()
    conf_correct  = probs_correct.max(axis=1)
    pred_correct  = probs_correct.argmax(axis=1)
    acc_correct   = (pred_correct == all_labels).mean()
    ece_correct, bin_info_correct = expected_calibration_error(
        conf_correct, pred_correct, all_labels)

    # ── Strategy B: INCORRECT ────────────────────────────────────────────────
    # Apply softmax over ALL logits (real + virtual).  The probability mass
    # that flows to virtual nodes is "stolen" from the real classes, so:
    #   • the maximum real-class probability is lower (under-confident)
    #   • accuracy drops because argmax may now point to a virtual node
    #     (which has no true label counterpart) — every such prediction is wrong
    all_logits_temp = all_logits / args.temp                   # (N, C_total)
    probs_wrong_all = F.softmax(all_logits_temp, dim=1).numpy()

    # Sub-case B1: argmax over ALL columns (prediction can fall into virtual space)
    pred_wrong_full  = probs_wrong_all.argmax(axis=1)
    conf_wrong_full  = probs_wrong_all.max(axis=1)
    # A prediction is correct only if it equals the true label AND is a real class
    correct_wrong_full = (pred_wrong_full == all_labels) & (pred_wrong_full < C_real)
    acc_wrong_full  = correct_wrong_full.mean()
    # For ECE, use only the real-class slice of probabilities (max of those)
    real_probs_renorm_full = probs_wrong_all[:, :C_real]
    conf_ece_full   = real_probs_renorm_full.max(axis=1)
    pred_ece_full   = real_probs_renorm_full.argmax(axis=1)
    ece_wrong_full, bin_info_wrong_full = expected_calibration_error(
        conf_ece_full, pred_ece_full, all_labels)

    # Sub-case B2: softmax over all, but restrict argmax to real classes
    # (common mistake: forget to re-normalise after the joint softmax)
    real_probs_from_joint = probs_wrong_all[:, :C_real]          # NOT renormalised
    pred_wrong_real  = real_probs_from_joint.argmax(axis=1)
    conf_wrong_real  = real_probs_from_joint.max(axis=1)         # deflated confidences
    acc_wrong_real   = (pred_wrong_real == all_labels).mean()
    ece_wrong_real, bin_info_wrong_real = expected_calibration_error(
        conf_wrong_real, pred_wrong_real, all_labels)

    # ── Compute extra diagnostics ─────────────────────────────────────────────
    # How much probability mass is stolen by virtual classes on average?
    virtual_mass_mean = probs_wrong_all[:, C_real:].sum(axis=1).mean()
    # How often does the argmax land in virtual space?
    virtual_pred_pct  = (pred_wrong_full >= C_real).mean()

    return {
        'correct':         {'acc': acc_correct,    'ece': ece_correct,
                            'bin_info': bin_info_correct,
                            'avg_max_conf': conf_correct.mean()},
        'wrong_full':      {'acc': acc_wrong_full,  'ece': ece_wrong_full,
                            'bin_info': bin_info_wrong_full,
                            'avg_max_conf': conf_ece_full.mean()},
        'wrong_real_only': {'acc': acc_wrong_real,  'ece': ece_wrong_real,
                            'bin_info': bin_info_wrong_real,
                            'avg_max_conf': conf_wrong_real.mean()},
        'virtual_mass_mean': virtual_mass_mean,
        'virtual_pred_pct':  virtual_pred_pct,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print & logging
# ─────────────────────────────────────────────────────────────────────────────
def print_and_log(results):
    lines = []
    SEP = "=" * 80

    lines.append("\n" + SEP)
    lines.append("  Q5 EXPERIMENT: Virtual Classes at Test Time")
    lines.append(SEP)
    lines.append(f"  Dataset          : {args.dataset}")
    lines.append(f"  Model            : {args.model_name}")
    lines.append(f"  Virtual Classes  : {args.v_classes}   (alpha = {args.alpha})")
    lines.append(f"  Temperature      : {args.temp}")
    lines.append(f"  Model file       : {args.model_file}")
    lines.append(f"  Max epoch trained: {args.final_epoch}")
    lines.append(SEP)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    lines.append("\n  [Diagnostic] When softmax is computed over ALL classes:")
    lines.append(f"    Average probability mass absorbed by virtual nodes : "
                 f"{results['virtual_mass_mean']:.4f}  "
                 f"({results['virtual_mass_mean']*100:.2f}%)")
    lines.append(f"    Fraction of samples whose argmax falls in virtual space: "
                 f"{results['virtual_pred_pct']:.4f}  "
                 f"({results['virtual_pred_pct']*100:.2f}%)")

    # ── Summary table ────────────────────────────────────────────────────────
    lines.append("\n" + "-" * 80)
    lines.append(f"  {'Strategy':<45} {'Accuracy':>9} {'ECE':>8} {'Avg MaxConf':>12}")
    lines.append("-" * 80)

    r = results['correct']
    lines.append(f"  {'(A) CORRECT  – softmax on real logits only':<45} "
                 f"{r['acc']:>9.4f} {r['ece']:>8.4f} {r['avg_max_conf']:>12.4f}")

    r = results['wrong_real_only']
    lines.append(f"  {'(B1) WRONG   – joint softmax, argmax on real cols':<45} "
                 f"{r['acc']:>9.4f} {r['ece']:>8.4f} {r['avg_max_conf']:>12.4f}")

    r = results['wrong_full']
    lines.append(f"  {'(B2) WRONG   – joint softmax, argmax over all cols':<45} "
                 f"{r['acc']:>9.4f} {r['ece']:>8.4f} {r['avg_max_conf']:>12.4f}")

    lines.append("-" * 80)

    # ── Delta summary ────────────────────────────────────────────────────────
    acc_a   = results['correct']['acc']
    ece_a   = results['correct']['ece']
    conf_a  = results['correct']['avg_max_conf']

    for tag, key in [('B1', 'wrong_real_only'), ('B2', 'wrong_full')]:
        r = results[key]
        da = r['acc']        - acc_a
        de = r['ece']        - ece_a
        dc = r['avg_max_conf'] - conf_a
        lines.append(f"\n  Delta (A → {tag}):")
        lines.append(f"    Accuracy change   : {da:+.4f}  ({da*100:+.2f} pp)")
        lines.append(f"    ECE change        : {de:+.4f}  (positive = worse calibration)")
        lines.append(f"    Avg max-conf change: {dc:+.4f}")

    # ── Per-bin detail for strategy A vs B1 ──────────────────────────────────
    lines.append("\n" + SEP)
    lines.append("  Per-Bin Detail  |  (A) Correct  vs  (B1) Wrong-real-only")
    lines.append(SEP)
    lines.append(f"  {'Bin':<12} {'N(A)':>6} {'Acc(A)':>8} {'Conf(A)':>8}"
                 f"  |  {'N(B1)':>6} {'Acc(B1)':>8} {'Conf(B1)':>8}")
    lines.append("-" * 80)

    bw = results['correct']['bin_info']['bin_width']
    M  = results['correct']['bin_info']['num']
    for i in range(M):
        bl = round(i * bw, 10)
        bu = round(bl + bw, 10)
        label = f"  {bl:.1f}–{bu:.1f}"
        ba = results['correct']['bin_info'].get(bl, {})
        bb = results['wrong_real_only']['bin_info'].get(bl, {})
        na = ba.get('num', 0)
        nb = bb.get('num', 0)
        acc_a_str  = f"{ba['acc']:.4f}"  if 'acc'      in ba else "—"
        conf_a_str = f"{ba['avg_conf']:.4f}" if 'avg_conf' in ba else "—"
        acc_b_str  = f"{bb['acc']:.4f}"  if 'acc'      in bb else "—"
        conf_b_str = f"{bb['avg_conf']:.4f}" if 'avg_conf' in bb else "—"
        lines.append(f"  {label:<12} {na:>6} {acc_a_str:>8} {conf_a_str:>8}"
                     f"  |  {nb:>6} {acc_b_str:>8} {conf_b_str:>8}")

    lines.append(SEP + "\n")

    # ── Discussion ────────────────────────────────────────────────────────────
    lines.append("  DISCUSSION")
    lines.append("  " + "-" * 76)
    lines.append(textwrap.fill(
        "Virtual smoothing classes are introduced during *training* to act as "
        "label-smoothing surrogates: the model is asked to spread a small fraction "
        "(alpha) of its total confidence across these K virtual nodes, which "
        "prevents over-confidence on real classes. However, the virtual nodes have "
        "NO corresponding ground-truth labels in the test set; they are purely "
        "a training device.", width=78, initial_indent="  ", subsequent_indent="  "))
    lines.append("")
    lines.append(textwrap.fill(
        "Strategy (A) — CORRECT: only the real-class logits are extracted, "
        "temperature-scaled, and normalised.  The virtual logits are discarded "
        "entirely, so the resulting distribution sums to 1 over the C real classes "
        "and directly reflects the model's calibrated belief.", width=78,
        initial_indent="  ", subsequent_indent="  "))
    lines.append("")
    lines.append(textwrap.fill(
        "Strategy (B1) — WRONG: softmax is computed jointly over real + virtual "
        "logits. Even when argmax is restricted to real classes, the probability "
        "mass is deflated because virtual nodes absorb part of it. Confidences are "
        "artificially low, which worsens ECE (the model now looks under-confident "
        "rather than well-calibrated) and can change which real class wins the "
        "argmax, reducing accuracy.", width=78,
        initial_indent="  ", subsequent_indent="  "))
    lines.append("")
    lines.append(textwrap.fill(
        "Strategy (B2) — WRONG: same joint softmax, but argmax is taken over all "
        "C + K columns. Predictions can now fall into the virtual space (none of "
        "which match any true label), so accuracy drops by the fraction of test "
        "samples whose top-1 logit is a virtual node.  Because the model was "
        "explicitly trained to route alpha probability to virtual nodes, this "
        "fraction is non-trivial even with small alpha.", width=78,
        initial_indent="  ", subsequent_indent="  "))
    lines.append("\n" + SEP + "\n")

    for line in lines:
        print(line)

    log_path = os.path.join(args.testing_logs, "q5_virtual_class_experiment.txt")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  [LOG] Results saved to {log_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────
def get_test_loader():
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    if args.dataset == 'cifar10':
        ds = datasets.CIFAR10('../../datasets/cifar10', train=False, download=True,
                              transform=T.ToTensor())
    elif args.dataset == 'cifar100':
        ds = datasets.CIFAR100('../../datasets/cifar100', train=False, download=True,
                               transform=T.ToTensor())
    elif args.dataset == 'svhn':
        ds = torchvision.datasets.SVHN(root='../../datasets/svhn', split='test',
                                       download=True,
                                       transform=T.Compose([T.ToTensor(),
                                           T.Normalize([0.5]*3, [0.5]*3)]))
    elif args.dataset == 'tiny-imagenet-64x64':
        normalize = T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        ds = torchvision.datasets.ImageFolder(
            root='../../datasets/tiny-imagenet-200/val',
            transform=T.Compose([T.ToTensor(), normalize]))
    elif args.dataset == 'imagenet':
        _, loader = imagenet_loader.data_loader('../../datasets/', batch_size=args.batch_size)
        return loader
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')
    return torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                       shuffle=False, **kwargs)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
import textwrap

def main():
    print('=' * 80)
    print('  Q5 Experiment: Correct vs Incorrect Virtual-Class Prediction')
    print(f'  args: {args}')
    print('=' * 80)

    test_loader = get_test_loader()
    model = get_model(args.model_name,
                      num_real_classes=NUM_REAL_CLASSES,
                      num_v_classes=args.v_classes,
                      dataset=args.dataset)
    cpt = filter_state_dict(torch.load(args.model_file, map_location='cpu'))
    model.load_state_dict(cpt)
    model = model.to(device)
    model.eval()

    results = run_experiment(model, test_loader)
    print_and_log(results)


if __name__ == '__main__':
    main()