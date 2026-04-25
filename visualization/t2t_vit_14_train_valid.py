import re
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.join(SCRIPT_DIR, "..", "training_logs", "mnist")

LOG_PATHS = {
    "1-channel (t2t_vit_14_1D)": os.path.join(BASE_DIR, "t2t_vit_14_1D", "log.txt"),
    "3-channel (t2t_vit_14_3D)": os.path.join(BASE_DIR, "t2t_vit_14_3D", "log.txt"),
}

def parse_log(filepath):
    epochs, test_acc, train_loss = [], [], []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"(?=Epoch:\s*\d+)", content)

    for block in blocks:
        epoch_match = re.search(r"Epoch:\s*(\d+)", block)
        if not epoch_match:
            continue

        epoch = int(epoch_match.group(1))

        acc_match   = re.search(r"Test Nat Acc\s*\|\s*([\d.]+)", block)
        loss_match  = re.search(r"Train Nat Acc\s*\|\s*([\d.]+)", block)

        if acc_match and loss_match:
            epochs.append(epoch)
            test_acc.append(float(acc_match.group(1)))
            train_loss.append(float(loss_match.group(1)))

    return epochs, test_acc, train_loss


def plot_graphs(log_paths, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    colors     = ["#378ADD", "#D85A30"]
    linestyles = ["-",       "--"]
    markers    = ["o",       "s"]

    all_data = {}
    for label, path in log_paths.items():
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path} — skipping.")
            continue
        epochs, test_acc, train_acc = parse_log(path)
        if not epochs:
            print(f"[WARNING] No data parsed from: {path}")
            continue
        all_data[label] = (epochs,train_acc ,test_acc)

    if not all_data:
        print("[ERROR] No data to plot. Check your log file paths.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("T2T-ViT-14 Training Comparison", fontsize=14, fontweight="bold", y=1.02)

    for idx, (label, (epochs, train_acc, test_acc)) in enumerate(all_data.items()):
        c  = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]
        mk = markers[idx % len(markers)]

        # Graph 1 — Train Accuracy
        axes[1].plot(epochs, train_acc, label=label, color=c,
                     linestyle=ls, marker=mk, linewidth=2, markersize=5)
        # Graph 2 — Validation Accuracy
        axes[0].plot(epochs, test_acc, label=label, color=c,
                     linestyle=ls, marker=mk, linewidth=2, markersize=5)

    for ax, title, ylabel in zip(
        axes,
        ["Train Accuracy vs Epoch", "Validation Accuracy vs Epoch"],
        ["Train Accuracy (%)", "Validation Accuracy (%)"],
    ):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "t2t_vit_14_variants_comparison_train_valid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_graphs(LOG_PATHS, output_dir="results/t2t_vit_14_variants")