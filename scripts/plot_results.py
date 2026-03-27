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

    # Sort by bitwidth for Pareto line
    points.sort(key=lambda p: p[0])

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Pareto frontier line
    ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        color="#cccccc", linewidth=1, linestyle="--", zorder=1,
    )

    # Points coloured by strategy
    plotted = set()
    for bw, lat, strat in points:
        color = STRATEGY_COLOR.get(strat, "#999999")
        marker = STRATEGY_MARKER.get(strat, "o")
        label = STRATEGY_LABEL.get(strat, strat) if strat not in plotted else None
        ax.scatter(bw, lat, color=color, marker=marker, s=80, zorder=3, label=label)
        plotted.add(strat)
        ax.annotate(
            f"{bw}-bit",
            xy=(bw, lat), xytext=(4, 4), textcoords="offset points", fontsize=7,
        )

    ax.set_xlabel("Avg Bitwidth")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("FP16 TinyLlama Pareto Front (NAS)")
    ax.legend(loc="upper right")

    save_fig(fig, out_dir, "fig1_pareto_scatter")


# ---------------------------------------------------------------------------
# Plot 2 — Sequence length scaling log-log
# ---------------------------------------------------------------------------

def plot_seqlen_scaling(benchmark_dir: Path, out_dir: Path):
    data = load_json(benchmark_dir / "benchmark_tinyllama.json")
    if data is None:
        return

    seqlen_data = data.get("seqlen_scaling", {})
    strategies = ["baseline", "int8_flex", "int8_rmsnorm"]

    # Collect valid (seq_len, latency) pairs per strategy
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

    for strat in strategies:
        xs, ys = series[strat]
        if not xs:
            continue
        ax.plot(
            xs, ys,
            color=STRATEGY_COLOR.get(strat, "#999999"),
            marker=STRATEGY_MARKER.get(strat, "o"),
            markersize=4,
            linewidth=1.2,
            label=STRATEGY_LABEL.get(strat, strat),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Seq-Length Scaling — TinyLlama (FP32, batch=1)")
    ax.legend(loc="upper left")

    save_fig(fig, out_dir, "fig2_seqlen_scaling")


# ---------------------------------------------------------------------------
# Plot 3 — Peak GPU memory vs sequence length
# ---------------------------------------------------------------------------

def plot_memory_scaling(benchmark_dir: Path, out_dir: Path):
    data = load_json(benchmark_dir / "benchmark_tinyllama.json")
    if data is None:
        return

    seqlen_data = data.get("seqlen_scaling", {})
    strategies = ["baseline", "int8_flex"]

    series: dict[str, tuple[list, list]] = {s: ([], []) for s in strategies}
    for seq_str, strat_dict in sorted(seqlen_data.items(), key=lambda x: int(x[0])):
        seq = int(seq_str)
        for strat in strategies:
            val = strat_dict.get(strat, {})
            mem = val.get("peak_memory_mb") if isinstance(val, dict) else None
            if mem is not None:
                series[strat][0].append(seq)
                series[strat][1].append(mem)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for strat in strategies:
        xs, ys = series[strat]
        if not xs:
            continue
        ax.plot(
            xs, ys,
            color=STRATEGY_COLOR.get(strat, "#999999"),
            marker=STRATEGY_MARKER.get(strat, "o"),
            markersize=4,
            linewidth=1.2,
            label=STRATEGY_LABEL.get(strat, strat),
        )

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("Peak Memory vs Seq-Length — TinyLlama")
    ax.legend(loc="upper left")

    save_fig(fig, out_dir, "fig3_memory_scaling")


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
    plot_memory_scaling(args.benchmark_dir, args.out_dir)
    print(f"Done. Figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
