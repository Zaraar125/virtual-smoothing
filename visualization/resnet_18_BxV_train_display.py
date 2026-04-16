import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
parser = argparse.ArgumentParser(description='Generating ResNet-18 BxV plots')
parser.add_argument('--log_root', type=str, default="testing_logs/cifar_10")
parser.add_argument('--output_dir', type=str, default="results/resnet_18_BxV")
args = parser.parse_args()

# ── Global Style (Publication Quality) ─────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Config ────────────────────────────────────────────────────────────────────
LOG_ROOT = os.path.join(PROJECT_ROOT,args.log_root)
OUTPUT_DIR = os.path.join(PROJECT_ROOT,args.output_dir)

os.makedirs(OUTPUT_DIR, exist_ok=True)
FOLDERS = [i for i in os.listdir(LOG_ROOT) if i.startswith("resnet_18")]

def log_parse_train(path):
    """
    Parses training log and extracts:
    - epoch numbers
    - train accuracy
    - test (validation) accuracy
    """
    text = open(path).read()

    epochs = []
    train_acc = []
    test_acc = []

    # Split by epoch blocks
    blocks = re.split(r"=+\nEpoch:\s*(\d+)", text)

    # blocks format:
    # [junk, epoch1, content1, epoch2, content2, ...]
    for i in range(1, len(blocks), 2):
        epoch = int(blocks[i])
        content = blocks[i+1]

        train_m = re.search(r"Train Nat Acc\s*\|\s*([\d.]+)", content)
        test_m  = re.search(r"Test Nat Acc\s*\|\s*([\d.]+)", content)

        if train_m and test_m:
            epochs.append(epoch)
            train_acc.append(float(train_m.group(1)))
            test_acc.append(float(test_m.group(1)))

    return epochs, train_acc, test_acc



# ── Training Curve Visualization ──────────────────────────────────────────────


train_records = []

for folder in FOLDERS:
    log_path = os.path.join(LOG_ROOT, folder, "log.txt")

    if not os.path.exists(log_path):
        print(f"[WARN] missing: {log_path}")
        continue

    epochs, train_acc, test_acc = log_parse_train(log_path)

    is_base = (folder == "resnet_18")

    train_records.append(dict(
        label=folder,
        epochs=epochs,
        train_acc=train_acc,
        test_acc=test_acc,
        is_base=is_base
    ))

# ── Colors (consistent with earlier) ──────────────────────────────────────────
cmap = plt.cm.get_cmap("tab10", len(train_records))

colors = []
for i, r in enumerate(train_records):
    if r["is_base"]:
        colors.append("#D62728")
    else:
        colors.append(cmap(i))

# ── Combined Plot: Training + Validation ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

# ── Left: Training Accuracy ───────────────────────────────────────────────────
ax = axes[0]

for i, r in enumerate(train_records):
    lw = 2.5 if r["is_base"] else 1.6
    alpha = 1.0 if r["is_base"] else 0.85

    ax.plot(r["epochs"],
            np.array(r["train_acc"]) / 100.0,
            label=r["label"],
            color=colors[i],
            linewidth=lw,
            alpha=alpha)

ax.set_xlabel("Epoch")
ax.set_ylabel("Training Accuracy")
ax.set_title("Training Accuracy vs Epoch")

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v:.0%}")
)

ax.grid(True, linestyle="--", alpha=0.3)


# ── Right: Validation Accuracy ────────────────────────────────────────────────
ax = axes[1]

for i, r in enumerate(train_records):
    lw = 2.5 if r["is_base"] else 1.6
    alpha = 1.0 if r["is_base"] else 0.85

    ax.plot(r["epochs"],
            np.array(r["test_acc"]) / 100.0,
            label=r["label"],
            color=colors[i],
            linewidth=lw,
            alpha=alpha)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy")
ax.set_title("Validation Accuracy vs Epoch")

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v:.0%}")
)

ax.grid(True, linestyle="--", alpha=0.3)


# ── Shared Legend (Clean) ─────────────────────────────────────────────────────
handles = [
    plt.Line2D([0], [0],
               color=colors[i],
               lw=2.5 if r["is_base"] else 1.6,
               label=r["label"])
    for i, r in enumerate(train_records)
]

fig.legend(handles=handles,
           loc="center left",
           bbox_to_anchor=(1.02, 0.5),
           frameon=False,
           title="Models")


# ── Layout & Save ─────────────────────────────────────────────────────────────
plt.tight_layout()

combined_path = os.path.join(OUTPUT_DIR, "train_val_accuracy_combined.png")
plt.savefig(combined_path, bbox_inches="tight")

print(f"Saved → {combined_path}")
# plt.show()