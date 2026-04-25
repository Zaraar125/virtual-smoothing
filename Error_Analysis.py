"""
Error Analysis Script – ResNet-18 on CIFAR-10
==============================================
Runs inference on the CIFAR-10 test set, collects the first 20
misclassified samples, and saves each one as:

    Error_Analysis/<true_label>x<pred_label>_<idx>.png

Usage
-----
python error_analysis_eval.py \
    --model_file ./dnn_models/cifar10/model.pt \
    [--v_classes 0] \
    [--alpha 0.0] \
    [--batch_size 128] \
    [--gpuid 0] \
    [--no_cuda]
"""

from __future__ import print_function

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision import datasets

# ── local imports (same repo structure as the original file) ─────────────────
from models import resnet
# ─────────────────────────────────────────────────────────────────────────────

# ── PIL is used only for saving images ───────────────────────────────────────
from PIL import Image

# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser(description='Error Analysis – ResNet-18 / CIFAR-10')
parser.add_argument('--model_file', default='./dnn_models/cifar10/model.pt',
                    help='Path to the saved model checkpoint (.pt)')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--no_cuda',   action='store_true', default=False)
parser.add_argument('--gpuid',     type=int, default=0)
parser.add_argument('--seed',      type=int, default=1)
parser.add_argument('--v_classes', type=int, default=0,
                    help='Number of virtual smoothing classes used during training')
parser.add_argument('--alpha',     type=float, default=0.0,
                    help='Total confidence of virtual smoothing classes')
parser.add_argument('--temp',      type=float, default=1.0,
                    help='Temperature scaling factor')
parser.add_argument('--output_dir', default='Error_Analysis',
                    help='Directory where misclassified images are saved')
parser.add_argument('--num_errors', type=int, default=20,
                    help='How many misclassified samples to save')
args = parser.parse_args()

# =============================================================================
# Constants
# =============================================================================
NUM_REAL_CLASSES = 10
CIFAR10_CLASSES  = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# =============================================================================
# Device setup
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
torch.manual_seed(args.seed)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(f"[INFO] Using device: {device}")

# =============================================================================
# Helper – strip DataParallel / 'module.' prefixes from state-dict keys
# =============================================================================
def filter_state_dict(state_dict):
    from collections import OrderedDict
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        new_sd[k[7:] if k.startswith('module.') else k] = v
    return new_sd

# =============================================================================
# Main
# =============================================================================
def main():
    # ── Data loader (no normalisation – we need the raw pixel values for saving)
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # We apply ToTensor() for the model but keep a *un-normalised* copy for
    # saving.  CIFAR-10 baseline models are often trained without per-channel
    # normalisation, so ToTensor() alone is used here – matching the original
    # eval script.
    transform_test = T.Compose([T.ToTensor()])

    test_dataset = datasets.CIFAR10(
        root='../../datasets/cifar10',
        train=False, download=True,
        transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = resnet.ResNet18(
        num_real_classes=NUM_REAL_CLASSES,
        num_v_classes=args.v_classes,
        normalizer=None
    )
    checkpoint = torch.load(args.model_file, map_location='cpu')
    model.load_state_dict(filter_state_dict(checkpoint))
    model = model.to(device)
    model.eval()
    print(f"[INFO] Model loaded from: {args.model_file}")

    # ── Output directory ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Inference + error collection ─────────────────────────────────────────
    misclassified = []          # list of (image_tensor, true_label, pred_label, global_idx)
    global_idx    = 0           # absolute sample index across all batches

    print(f"[INFO] Scanning test set for misclassified samples …")

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            if len(misclassified) >= args.num_errors:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            # Slice only the real classes, apply temperature scaling
            probs  = F.softmax(logits[:, :NUM_REAL_CLASSES] / args.temp, dim=1)

            # Optional virtual-class re-normalisation (mirrors original cal_ece)
            if args.alpha > 0 and args.v_classes == 0:
                mins = probs.min(dim=1)[0]
                for i in range(len(probs)):
                    probs[i] = probs[i] - mins[i]
                    probs[i] = probs[i] / probs[i].sum()

            preds = probs.argmax(dim=1)           # predicted class index

            # Iterate sample-by-sample within the batch
            for j in range(len(batch_y)):
                true_lbl = batch_y[j].item()
                pred_lbl = preds[j].item()

                if true_lbl != pred_lbl:
                    # Store CPU tensor (CHW, float [0,1])
                    misclassified.append((
                        batch_x[j].cpu(),
                        true_lbl,
                        pred_lbl,
                        global_idx + j
                    ))
                    if len(misclassified) >= args.num_errors:
                        break

            global_idx += len(batch_y)

    print(f"[INFO] Found {len(misclassified)} misclassified samples.")

    # ── Save images ───────────────────────────────────────────────────────────
    # Track how many times each (true, pred) pair has appeared so filenames
    # stay unique even when the same confusion repeats.
    pair_count = {}

    saved_info = []   # for the summary table printed at the end

    for img_tensor, true_lbl, pred_lbl, sample_idx in misclassified:
        true_name = CIFAR10_CLASSES[true_lbl]
        pred_name = CIFAR10_CLASSES[pred_lbl]

        pair_key = (true_name, pred_name)
        pair_count[pair_key] = pair_count.get(pair_key, 0) + 1
        count = pair_count[pair_key]

        # Filename: orig_labelxpred_label[_N].png  (N added from the 2nd duplicate)
        if count == 1:
            filename = f"{true_name}x{pred_name}.png"
        else:
            filename = f"{true_name}x{pred_name}_{count}.png"

        filepath = os.path.join(args.output_dir, filename)

        # Convert CHW float tensor → HWC uint8 numpy → PIL Image
        img_np  = (img_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Upscale 32×32 → 128×128 for easier visual inspection
        pil_img = pil_img.resize((128, 128), Image.NEAREST)
        pil_img.save(filepath)

        saved_info.append((filename, true_name, pred_name, sample_idx))

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  Error Analysis – {len(saved_info)} Misclassified Samples Saved")
    print(f"  Output directory: {os.path.abspath(args.output_dir)}")
    print("=" * 65)
    print(f"  {'#':<4} {'File':<35} {'True':>12} {'Predicted':>12}")
    print("-" * 65)
    for i, (fname, true_name, pred_name, sidx) in enumerate(saved_info, 1):
        print(f"  {i:<4} {fname:<35} {true_name:>12} {pred_name:>12}  (sample #{sidx})")
    print("=" * 65)
    print()


if __name__ == '__main__':
    main()