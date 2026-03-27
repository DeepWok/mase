"""
Part 3 Report — Figure Generation

Produces three publication-quality plots:
  fig1_pareto_scatter.png   — Bitwidth vs latency Pareto front (FP16 TinyLlama)
  fig2_seqlen_scaling.png   — Latency vs sequence length log-log (TinyLlama FP32)
  fig3_memory_scaling.png   — Peak GPU memory vs sequence length (TinyLlama FP32)

Usage (run on cluster after git pull):
    ~/mase/.venv/bin/python scripts/plot_results.py \
        --benchmark-dir outputs/benchmark/ \
        --profiling-dir outputs/profiling/ \
        --search-dir    outputs/search/llama_20260327_1557/ \
        --out-dir       outputs/figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Consistent colours for fusion strategies across all plots
STRATEGY_COLOR = {
    "baseline":      "#555555",
    "none":          "#555555",
    "int8_none":     "#aaaaaa",
    "fused_rmsnorm": "#1f77b4",
    "int8_rmsnorm":  "#1f77b4",
    "flex_attention":"#e6550d",
    "int8_flex":     "#e6550d",
    "both":          "#2ca02c",
    "int8_both":     "#2ca02c",
}

STRATEGY_LABEL = {
    "baseline":      "Baseline",
    "none":          "No fusion",
    "int8_none":     "INT8 only",
    "fused_rmsnorm": "FusedRMSNorm",
    "int8_rmsnorm":  "INT8 + FusedRMSNorm",
    "flex_attention":"FlexAttention",
    "int8_flex":     "INT8 + FlexAttention",
    "both":          "Both fused",
    "int8_both":     "INT8 + Both",
}

STRATEGY_MARKER = {
    "baseline": "o", "none": "o",
    "int8_none": "s",
    "fused_rmsnorm": "^", "int8_rmsnorm": "^",
    "flex_attention": "D", "int8_flex": "D",
    "both": "*", "int8_both": "*",
}

FIG_SIZE = (3.5, 2.6)  # IEEE single-column


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}.png / .pdf")


# ---------------------------------------------------------------------------
# Plot 1 — Pareto scatter: bitwidth vs latency (FP16 TinyLlama search)
# ---------------------------------------------------------------------------

def plot_pareto_scatter(search_dir: Path, out_dir: Path):
    data = load_json(search_dir / "best.json")
    if data is None:
        return

    points = []
    for entry in data.values():
        hw = entry.get("hardware_metrics", {})
        cfg = entry.get("sampled_config", {})
        bw = hw.get("average_bitwidth")
        lat = hw.get("latency")
        strategy = cfg.get("fusion_strategy", "none")
        if bw is not None and lat is not None:
            points.append((bw, lat, strategy))

    # Sort by latency descending for a clean Pareto frontier line
    # (higher latency = worse, so Pareto front goes top-left to bottom-right)
    points.sort(key=lambda p: p[1], reverse=True)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Pareto frontier line through all points sorted by latency
    ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        color="#cccccc", linewidth=1, linestyle="--", zorder=1,
    )

    # Points coloured by strategy — label each point with bitwidth + strategy
    plotted = set()
    offsets = {0: (5, 5), 1: (5, -12), 2: (5, 5), 3: (-38, 5)}
    for i, (bw, lat, strat) in enumerate(points):
        color = STRATEGY_COLOR.get(strat, "#999999")
        marker = STRATEGY_MARKER.get(strat, "o")
        label = STRATEGY_LABEL.get(strat, strat) if strat not in plotted else None
        ax.scatter(bw, lat, color=color, marker=marker, s=90, zorder=3, label=label)
        plotted.add(strat)
        dx, dy = offsets.get(i, (5, 5))
        ax.annotate(
            f"{int(bw)}-bit\n({STRATEGY_LABEL.get(strat, strat)})",
            xy=(bw, lat), xytext=(dx, dy), textcoords="offset points",
            fontsize=6.5, color=color,
        )

    ax.set_xlabel("Avg Bitwidth")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("FP16 TinyLlama Pareto Front (NAS)")
    ax.legend(loc="upper right")

    save_fig(fig, out_dir, "fig1_pareto_scatter")


# ---------------------------------------------------------------------------
# Plot 2 — Sequence length scaling log-log (Mistral sliding-window)
# Mistral uses sliding-window FlexAttention → O(n·w) attention compute,
# demonstrating sub-quadratic scaling vs baseline O(n²) SDPA.
# ---------------------------------------------------------------------------

def plot_seqlen_scaling(benchmark_dir: Path, out_dir: Path):
    data = load_json(benchmark_dir / "benchmark_mistral.json")
    if data is None:
        return

    seqlen_data = data.get("seqlen_scaling", {})
    strategies = ["baseline", "int8_flex", "int8_none"]

    series: dict[str, tuple[list, list]] = {s: ([], []) for s in strategies}
    for seq_str, strat_dict in sorted(seqlen_data.items(), key=lambda x: int(x[0])):
        seq = int(seq_str)
        for strat in strategies:
            val = strat_dict.get(strat, {})
            lat = val.get("latency_ms") if isinstance(val, dict) else None
            if lat is not None:
                series[strat][0].append(seq)
                series[strat][1].append(lat)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    DISPLAY = {
        "baseline": ("Baseline (SDPA)", STRATEGY_COLOR["baseline"], "o"),
        "int8_none": ("INT8 + SDPA",     STRATEGY_COLOR["int8_none"], "s"),
        "int8_flex": ("INT8 + FlexAttn (SWA)", STRATEGY_COLOR["int8_flex"], "D"),
    }

    for strat, (label, color, marker) in DISPLAY.items():
        xs, ys = series[strat]
        if not xs:
            continue
        ax.plot(xs, ys, color=color, marker=marker, markersize=4,
                linewidth=1.2, label=label)

    # Reference slope lines anchored at seq=1024 to baseline latency there.
    # Finds the baseline value closest to seq=1024 as the anchor point.
    import numpy as np
    bxs, bys = series["baseline"]
    if len(bxs) >= 2:
        anchor_x = 1024
        anchor_y = np.interp(anchor_x, bxs, bys)
        ref_xs = np.array([min(bxs), max(bxs)], dtype=float)
        # O(n²): latency ∝ n²
        slope2 = anchor_y * (ref_xs / anchor_x) ** 2
        ax.plot(ref_xs, slope2, color="#bbbbbb", linewidth=0.9,
                linestyle=":", label=r"$O(n^2)$ ref")
        # O(n·w) with w=128: latency ∝ n (linear)
        slope1 = anchor_y * (ref_xs / anchor_x) ** 1
        ax.plot(ref_xs, slope1, color="#bbbbbb", linewidth=0.9,
                linestyle=(0, (3, 1, 1, 1)), label=r"$O(n{\cdot}w)$ ref")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Seq-Length Scaling — Mistral-7B (FP16)")
    ax.legend(loc="upper left", fontsize=7)

    save_fig(fig, out_dir, "fig2_seqlen_scaling")


# ---------------------------------------------------------------------------
# Plot 3 — Kernel dispatch bar chart (TinyLlama + Mistral)
# Replaces unreliable peak-memory plot. Shows fused_rmsnorm ~18% dispatch
# reduction; flex_attention contributes zero dispatch savings.
# ---------------------------------------------------------------------------

def plot_kernel_dispatch(profiling_dir: Path, out_dir: Path):
    import numpy as np

    models = [
        ("tinyllama", "TinyLlama-1.1B"),
        ("mistral",   "Mistral-7B"),
    ]
    strategies = ["baseline", "fused_rmsnorm", "flex_attention", "both"]
    strategy_labels = ["Baseline", "FusedRMSNorm", "FlexAttn", "Both"]
    colors = [
        STRATEGY_COLOR["baseline"],
        STRATEGY_COLOR["fused_rmsnorm"],
        STRATEGY_COLOR["flex_attention"],
        STRATEGY_COLOR["both"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=False)

    for ax, (model_key, model_label) in zip(axes, models):
        data = load_json(profiling_dir / f"kernel_profile_{model_key}_seq512.json")
        if data is None:
            continue
        counts = []
        for strat in strategies:
            s = data["strategies"].get(strat, {})
            counts.append(s.get("total_kernel_launches", 0))

        bars = ax.bar(strategy_labels, counts, color=colors, width=0.6, edgecolor="white")

        # Annotate reduction % vs baseline
        baseline = counts[0]
        for bar, count in zip(bars, counts):
            pct = 100 * (baseline - count) / baseline if baseline > 0 else 0
            label = f"{count:,}" if pct == 0 else f"{count:,}\n({pct:.0f}%↓)"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    label, ha="center", va="bottom", fontsize=6.5)

        ax.set_title(model_label, fontsize=9)
        ax.set_ylabel("Kernel Dispatches" if ax == axes[0] else "")
        ax.set_ylim(0, max(counts) * 1.22)
        ax.tick_params(axis="x", labelsize=7.5)

    fig.suptitle("CUDA Kernel Dispatches per Fusion Strategy (seq=512)", fontsize=9)
    plt.tight_layout()
    save_fig(fig, out_dir, "fig3_kernel_dispatch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--profiling-dir",  type=Path, default=Path("outputs/profiling"))
    parser.add_argument("--search-dir",     type=Path, default=Path("outputs/search/llama_20260327_1557"))
    parser.add_argument("--out-dir",        type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    print("Generating figures...")
    plot_pareto_scatter(args.search_dir, args.out_dir)
    plot_seqlen_scaling(args.benchmark_dir, args.out_dir)
    plot_kernel_dispatch(args.profiling_dir, args.out_dir)
    print(f"Done. Figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
