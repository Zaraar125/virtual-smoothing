import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
parser = argparse.ArgumentParser(description='Generating ResNet-18 BxV plots')
parser.add_argument('--log_root', type=str, default="training_logs/cifar_10")
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

# ── Parameter Counter ─────────────────────────────────────────────────────────
def count_params(base_width, num_blocks):
    def conv_params(Cin, Cout, k=3): return Cin * Cout * k * k
    def bn_params(C): return 2 * C

    total = 0
    w = base_width

    total += conv_params(3, w) + bn_params(w)

    widths = [w, w*2, w*4, w*8]
    in_ch = w

    for out_ch, nb in zip(widths, num_blocks):
        for block in range(nb):
            cin = in_ch if block == 0 else out_ch
            total += conv_params(cin, out_ch) + bn_params(out_ch)
            total += conv_params(out_ch, out_ch) + bn_params(out_ch)

            if block == 0 and cin != out_ch:
                total += conv_params(cin, out_ch, k=1) + bn_params(out_ch)

        in_ch = out_ch

    total += in_ch * 10 + 10
    return total / 1e6


# ── Log Parser ────────────────────────────────────────────────────────────────
def parse_log(path):
    text = open(path).read()

    acc_m = re.search(r"Overall Accuracy:\s+([\d.]+)", text)
    ece_m = re.search(r"Expected Calibration Error \(ECE\):\s+([\d.]+)", text)
    bw_m  = re.search(r"base_width:\s*(\d+)", text)
    nb_m  = re.search(r"num_blocks:\s*\[([0-9,\s]+)\]", text)

    acc = float(acc_m.group(1)) if acc_m else None
    ece = float(ece_m.group(1)) if ece_m else None

    if bw_m and nb_m:
        base_width = int(bw_m.group(1))
        num_blocks = [int(x) for x in nb_m.group(1).split(",")]
    else:
        base_width = 64
        num_blocks = [2, 2, 2, 2]

    params = count_params(base_width, num_blocks)
    return acc, ece, params, base_width, num_blocks


# ── Collect Data ──────────────────────────────────────────────────────────────
records = []
for folder in FOLDERS:
    log_path = os.path.join(LOG_ROOT, folder, "log.txt")
    if not os.path.exists(log_path):
        print(f"[WARN] missing: {log_path}")
        continue

    acc, ece, params, bw, nb = parse_log(log_path)
    is_base = (folder == "resnet_18")

    label = "Base" if is_base else f"w={bw}, b={nb}"
    records.append(dict(
        label=label,
        acc=acc,
        ece=ece,
        params=params,
        is_base=is_base
    ))

records.sort(key=lambda r: r["params"])

# ── Colors ────────────────────────────────────────────────────────────────────
cmap = plt.cm.get_cmap("tab10", len(records))
colors = []
for i, r in enumerate(records):
    if r["is_base"]:
        colors.append("#D62728")  # publication red
    else:
        colors.append(cmap(i))

# ── Prepare Data ──────────────────────────────────────────────────────────────
params_all = [r["params"] for r in records]
acc_all    = [r["acc"] for r in records]
ece_all    = [r["ece"] for r in records]

base = next(r for r in records if r["is_base"])

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
fig.patch.set_facecolor("white")

# ── Accuracy Plot ─────────────────────────────────────────────────────────────
ax = axes[0]

ax.plot(params_all, acc_all, linestyle="--", linewidth=1, alpha=0.5)

for i, r in enumerate(records):
    ax.scatter(r["params"], r["acc"],
               s=80,
               color=colors[i],
               edgecolor="black",
               linewidth=0.5,
               zorder=3)

    ax.annotate(r["label"],
                (r["params"], r["acc"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8)

ax.axhline(base["acc"], linestyle=":", linewidth=1,
           color="#D62728", alpha=0.7)

ax.set_xlabel("Parameters (Millions)")
ax.set_ylabel("Test Accuracy")
ax.set_title("Accuracy vs Model Size")

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: f"{v:.1%}")
)

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

# ── ECE Plot ──────────────────────────────────────────────────────────────────
ax = axes[1]

ax.plot(params_all, ece_all, linestyle="--", linewidth=1, alpha=0.5)

for i, r in enumerate(records):
    ax.scatter(r["params"], r["ece"],
               s=80,
               color=colors[i],
               edgecolor="black",
               linewidth=0.5,
               zorder=3)

    ax.annotate(r["label"],
                (r["params"], r["ece"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8)

ax.axhline(base["ece"], linestyle=":", linewidth=1,
           color="#D62728", alpha=0.7)

ax.set_xlabel("Parameters (Millions)")
ax.set_ylabel("ECE (↓ better)")
ax.set_title("Calibration vs Model Size")

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

# ── Legend (Minimal + Clean) ──────────────────────────────────────────────────
# ── Legend (All Variants) ─────────────────────────────────────────────────────
legend_elements = []

for i, r in enumerate(records):
    legend_elements.append(
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=r["label"],
            markerfacecolor=colors[i],
            markeredgecolor='black',
            markersize=7
        )
    )

# Add base reference line separately
legend_elements.append(
    plt.Line2D([0], [0],
               linestyle=":",
               color="#D62728",
               label="Base reference")
)

axes[0].legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),  # push legend outside
    frameon=False,
    title="Model Variants"
)
# ── Layout & Save ─────────────────────────────────────────────────────────────
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, "resnet18_BxV_test_plot.png")
plt.savefig(out_path, bbox_inches="tight")

print(f"Saved → {out_path}")





