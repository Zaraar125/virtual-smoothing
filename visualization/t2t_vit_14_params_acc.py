import re
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR   = os.path.join(SCRIPT_DIR, "..", "testing_logs", "mnist")

# Add more models here as needed
TEST_LOG_PATHS = {
    "t2t_vit_14\n(1-channel)": os.path.join(TEST_DIR, "t2t_vit_14_1D", "log.txt"),
    "t2t_vit_14\n(3-channel)": os.path.join(TEST_DIR, "t2t_vit_14_3D", "log.txt"),
}

# ---------------------------------------------------------------------------
# Analytical parameter counts
# Extend this dict when you add more models.
# Key must match the key in TEST_LOG_PATHS exactly.
# ---------------------------------------------------------------------------
def _token_performer_params(in_dim, out_dim=64):
    return in_dim * out_dim + out_dim * out_dim + in_dim * 2 + out_dim * 2

def _t2t_vit_params(in_chans, embed_dim, depth, mlp_ratio, num_classes):
    token_dim   = 64
    mlp_dim     = int(embed_dim * mlp_ratio)
    num_patches = (224 // (4 * 2 * 2)) ** 2   # 196

    t2t = (
        _token_performer_params(in_chans * 7 * 7, token_dim) +
        _token_performer_params(token_dim * 3 * 3, token_dim) +
        (token_dim * 3 * 3) * embed_dim + embed_dim
    )
    pos   = embed_dim + (num_patches + 1) * embed_dim   # cls_token + pos_embed
    block = (
        embed_dim * 2 + embed_dim * 2 +
        embed_dim * (embed_dim * 3) + embed_dim * 3 +
        embed_dim * embed_dim + embed_dim +
        embed_dim * mlp_dim + mlp_dim +
        mlp_dim * embed_dim + embed_dim
    ) * depth
    head  = embed_dim * 2 + embed_dim * num_classes + num_classes   # norm + head
    return (t2t + pos + block + head) / 1e6

MODEL_PARAMS = {
    "t2t_vit_14\n(1-channel)": _t2t_vit_params(in_chans=1, embed_dim=384, depth=14, mlp_ratio=3.0, num_classes=20),
    "t2t_vit_14\n(3-channel)": _t2t_vit_params(in_chans=3, embed_dim=384, depth=14, mlp_ratio=3.0, num_classes=20),
}

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_testing_log(filepath):
    """Return (overall_accuracy_pct, ece) from a testing/ECE log."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    acc_m = re.search(r"Overall Accuracy:\s+([\d.]+)", content)
    ece_m = re.search(r"Expected Calibration Error \(ECE\):\s+([\d.]+)", content)
    if not acc_m or not ece_m:
        return None, None
    return float(acc_m.group(1)) * 100, float(ece_m.group(1))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(test_log_paths, model_params, output_dir="."):
    colors  = ["#378ADD", "#D85A30", "#1D9E75", "#D4537E", "#BA7517"]
    markers = ["o", "s", "^", "D", "P"]

    records = []
    for idx, (label, path) in enumerate(test_log_paths.items()):
        if not os.path.exists(path):
            print(f"[WARNING] Not found: {path} — skipping.")
            continue
        acc, ece = parse_testing_log(path)
        if acc is None:
            print(f"[WARNING] Could not parse: {path}")
            continue
        params = model_params.get(label)
        if params is None:
            print(f"[WARNING] No parameter count defined for '{label}' — skipping.")
            continue
        records.append((label, params, acc, ece, idx))
        print(f"[INFO] {label.replace(chr(10), ' ')}: params={params:.2f}M  acc={acc:.2f}%  ECE={ece:.4f}")

    if not records:
        print("[ERROR] No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for label, params, acc, ece, idx in records:
        c  = colors[idx % len(colors)]
        mk = markers[idx % len(markers)]

        ax.scatter(params, acc, color=c, marker=mk, s=150, zorder=5)

        # Annotation: short label + ECE
        display = label.replace("\n", "  ")
        ax.annotate(
            f"{display}\nECE = {ece:.4f}",
            xy=(params, acc),
            xytext=(10, 14),
            textcoords="offset points",
            fontsize=9,
            color=c,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=c, lw=1.0, alpha=0.92),
            arrowprops=dict(arrowstyle="->", color=c, lw=0.9),
        )

    ax.set_title("Model Parameters vs Test Accuracy", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Parameters (Millions)", fontsize=11)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # X-axis: give breathing room around the points
    all_params = [r[1] for r in records]
    x_margin   = max(0.05, (max(all_params) - min(all_params)) * 0.5) if len(all_params) > 1 else 0.1
    ax.set_xlim(min(all_params) - x_margin, max(all_params) + x_margin)

    # Y-axis: start just below lowest accuracy
    all_acc  = [r[2] for r in records]
    y_margin = 0.5
    ax.set_ylim(min(all_acc) - y_margin, max(all_acc) + y_margin)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "params_vs_accuracy.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {out_path}")
    plt.show()


if __name__ == "__main__":
    plot(TEST_LOG_PATHS, MODEL_PARAMS, output_dir="results/t2t_vit_14_variants")