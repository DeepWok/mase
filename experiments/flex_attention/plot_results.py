#!/usr/bin/env python3
"""Generate publication-quality figures for FlexAttention experiment results."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

try:
    import seaborn as sns

    sns.set_style("whitegrid")
except ImportError:
    pass

# ── Paths ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Style ────────────────────────────────────────────────────────────────────
COLORS = {
    "SDPA causal": "#1f77b4",
    "SDPA SWA": "#6baed6",
    "Flex causal": "#e6550d",
    "Flex SWA": "#fdae6b",
    "SDPA doc": "#d62728",
    "Flex doc": "#2ca02c",
    "SDPA baseline": "#1f77b4",
}

MARKERS = {"SDPA": "o", "Flex": "^"}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"  WARNING: {name} not found, skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def save_fig(fig, name: str, png_only: bool = False):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    if not png_only:
        fig.savefig(FIGURES_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved {name}")


def extract_series(benchmarks: dict, method: str, metric: str = "mean_ms"):
    """Extract (seq_lens, values) from benchmarks dict for a given method."""
    data = benchmarks[method]
    seq_lens = sorted(data.keys(), key=int)
    values = [data[s][metric] for s in seq_lens]
    return [int(s) for s in seq_lens], values


def extract_series_with_std(benchmarks: dict, method: str):
    """Extract (seq_lens, means, stds) from benchmarks dict."""
    data = benchmarks[method]
    seq_lens = sorted(data.keys(), key=int)
    means = [data[s]["mean_ms"] for s in seq_lens]
    stds = [data[s].get("std_ms", 0) for s in seq_lens]
    return [int(s) for s in seq_lens], means, stds


# ── Figure 1: Inference Latency (exp1a) ─────────────────────────────────────
def fig1_inference_latency(png_only=False):
    d = load_json("exp1a_inference_latency_memory.json")
    if not d:
        return
    b = d["benchmarks"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    configs = [
        ("SDPA causal", "SDPA causal", COLORS["SDPA causal"], "o"),
        ("SDPA sliding_window(256)", "SDPA SWA(256)", COLORS["SDPA SWA"], "o"),
        ("Flex causal", "Flex causal", COLORS["Flex causal"], "^"),
        ("Flex sliding_window(256)", "Flex SWA(256)", COLORS["Flex SWA"], "^"),
    ]

    for key, label, color, marker in configs:
        seq, means, stds = extract_series_with_std(b, key)
        means, stds = np.array(means), np.array(stds)
        ax.plot(seq, means, marker=marker, label=label, color=color, linewidth=1.5, markersize=5)
        ax.fill_between(seq, means - stds, means + stds, alpha=0.15, color=color)

    # Annotate SWA speedup at 4096
    sdpa_swa = b["SDPA sliding_window(256)"]["4096"]["mean_ms"]
    flex_swa = b["Flex sliding_window(256)"]["4096"]["mean_ms"]
    speedup = sdpa_swa / flex_swa
    ax.annotate(
        f"{speedup:.2f}x",
        xy=(4096, flex_swa),
        xytext=(3200, flex_swa - 15),
        fontsize=9,
        fontweight="bold",
        color=COLORS["Flex SWA"],
        arrowprops=dict(arrowstyle="->", color=COLORS["Flex SWA"], lw=1.2),
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096])
    ax.set_xticklabels([256, 512, 1024, 2048, 4096])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency: SDPA vs FlexAttention")
    ax.legend(loc="upper left")
    try:
        sns.despine(ax=ax)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "fig1_inference_latency", png_only)


# ── Figure 2: Training Equivalence (exp2) ───────────────────────────────────
def fig2_training_equivalence(png_only=False):
    d = load_json("exp2_training_equivalence.json")
    if not d:
        return
    m = d["metrics"]

    losses_sdpa = np.array(m["losses_sdpa"])
    losses_flex = np.array(m["losses_flex"])
    steps = np.arange(1, len(losses_sdpa) + 1)
    diff = np.abs(losses_sdpa - losses_flex)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.5), height_ratios=[3, 1], sharex=True)

    ax1.plot(steps, losses_sdpa, label="SDPA (FA2)", color=COLORS["SDPA causal"], linewidth=1.5)
    ax1.plot(
        steps,
        losses_flex,
        label="FlexAttention",
        color=COLORS["Flex causal"],
        linewidth=1.5,
        linestyle="--",
    )
    ax1.set_ylabel("Training Loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training Equivalence: SDPA vs FlexAttention")

    ax2.semilogy(steps, diff, color="#333333", linewidth=1.2)
    ax2.axhline(m["max_loss_diff"], ls="--", color="gray", linewidth=0.8, label=f'max = {m["max_loss_diff"]:.2e}')
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("|Loss Diff|")
    ax2.legend(loc="upper right", fontsize=8)

    try:
        sns.despine(ax=ax1)
        sns.despine(ax=ax2)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "fig2_training_equivalence", png_only)


# ── Figure 3: Document Masking (exp8) ───────────────────────────────────────
def fig3_document_masking(png_only=False):
    d = load_json("exp8_document_masking.json")
    if not d:
        return
    b = d["benchmarks"]

    seq_lens = ["1024", "4096", "8192"]
    methods = [
        ("SDPA Causal (FA2 Baseline)", "SDPA Causal (FA2)", COLORS["SDPA causal"]),
        ("SDPA Document Mask", "SDPA Doc Mask", COLORS["SDPA doc"]),
        ("Flex Document Mask", "Flex Doc Mask", COLORS["Flex doc"]),
    ]

    x = np.arange(len(seq_lens))
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, (key, label, color) in enumerate(methods):
        vals = [b[key][s]["mean_ms"] for s in seq_lens]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, edgecolor="white")

        # Annotate speedup for Flex vs SDPA Doc Mask
        if key == "Flex Document Mask":
            for j, s in enumerate(seq_lens):
                sdpa_val = b["SDPA Document Mask"][s]["mean_ms"]
                flex_val = b[key][s]["mean_ms"]
                ratio = sdpa_val / flex_val
                ax.text(
                    x[j] + i * width,
                    flex_val + 20,
                    f"{ratio:.1f}x",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{int(s)}" for s in seq_lens])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Training Latency (ms)")
    ax.set_title("Document Masking: Sequence Packing Speedup")
    ax.legend(loc="upper left")
    try:
        sns.despine(ax=ax)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "fig3_document_masking", png_only)


# ── Figure 4: GQA Isolation Heatmap (exp12) ─────────────────────────────────
def fig4_gqa_heatmap(png_only=False):
    d = load_json("exp12_gqa_isolation.json")
    if not d:
        return
    b = d["benchmarks"]

    head_configs = ["MHA (16/16)", "GQA-4 (16/4)", "MQA (16/1)"]
    patterns = ["causal", "SWA(256)", "ALiBi+SWA(256)"]

    # Build speedup matrix at seq_len=4096
    matrix = np.zeros((3, 3))
    for i, hc in enumerate(head_configs):
        for j, pat in enumerate(patterns):
            sdpa_key = f"SDPA {hc} {pat}"
            flex_key = f"Flex {hc} {pat}"
            sdpa_ms = b[sdpa_key]["4096"]["mean_ms"]
            flex_ms = b[flex_key]["4096"]["mean_ms"]
            matrix[i, j] = sdpa_ms / flex_ms

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
    norm = mcolors.TwoSlopeNorm(vcenter=1.0, vmin=matrix.min() - 0.05, vmax=matrix.max() + 0.05)
    im = ax.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")

    # Cell annotations
    for i in range(3):
        for j in range(3):
            val = matrix[i, j]
            color = "white" if val > 1.4 else "black"
            ax.text(j, i, f"{val:.2f}x", ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(3))
    ax.set_xticklabels(patterns, fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels(head_configs, fontsize=9)
    ax.set_title("Speedup (SDPA / Flex) at seq_len=4096")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Speedup", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig4_gqa_heatmap", png_only)


# ── Figure 5: Decode Caveat (exp10) ─────────────────────────────────────────
def fig5_decode_caveat(png_only=False):
    d = load_json("exp10_decode_generation.json")
    if not d:
        return
    b = d["benchmarks"]

    prompt_lens = ["128", "512", "1024"]
    methods = [
        ("SDPA causal", "SDPA causal", COLORS["SDPA causal"]),
        ("Flex causal", "Flex causal", COLORS["Flex causal"]),
        ("Flex sliding_window(256)", "Flex SWA(256)", COLORS["Flex SWA"]),
    ]

    x = np.arange(len(prompt_lens))
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    for i, (key, label, color) in enumerate(methods):
        vals = [b[key][s]["per_token_ms"] for s in prompt_lens]
        ax.bar(x + i * width, vals, width, label=label, color=color, edgecolor="white")

        # Annotate slowdown vs SDPA
        if key != "SDPA causal":
            for j, s in enumerate(prompt_lens):
                sdpa_val = b["SDPA causal"][s]["per_token_ms"]
                flex_val = b[key][s]["per_token_ms"]
                ratio = flex_val / sdpa_val
                if ratio > 1.5:
                    ax.text(
                        x[j] + i * width,
                        flex_val + 1,
                        f"{ratio:.1f}x",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                        color=color,
                    )

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{int(s)}" for s in prompt_lens])
    ax.set_xlabel("Prompt Length")
    ax.set_ylabel("Per-Token Decode Latency (ms)")
    ax.set_title("Decode Generation: torch.compile Overhead")
    ax.legend(loc="upper left")
    try:
        sns.despine(ax=ax)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "fig5_decode_caveat", png_only)


# ── Figure 6: Training Throughput (exp11) ────────────────────────────────────
def fig6_training_throughput(png_only=False):
    d = load_json("exp11_throughput.json")
    if not d:
        return
    t = d["throughput"]["training"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    configs = [
        ("SDPA causal", "SDPA causal", COLORS["SDPA causal"], "o"),
        ("SDPA sliding_window(256)", "SDPA SWA(256)", COLORS["SDPA SWA"], "o"),
        ("Flex causal", "Flex causal", COLORS["Flex causal"], "^"),
        ("Flex sliding_window(256)", "Flex SWA(256)", COLORS["Flex SWA"], "^"),
    ]

    for key, label, color, marker in configs:
        seq, vals = extract_series(t, key, "throughput_tok_per_sec")
        ax.plot(seq, vals, marker=marker, label=label, color=color, linewidth=1.5, markersize=5)

    # Annotate SWA speedup at 4096
    sdpa_swa = t["SDPA sliding_window(256)"]["4096"]["throughput_tok_per_sec"]
    flex_swa = t["Flex sliding_window(256)"]["4096"]["throughput_tok_per_sec"]
    speedup = flex_swa / sdpa_swa
    ax.annotate(
        f"{speedup:.2f}x",
        xy=(4096, flex_swa),
        xytext=(3200, flex_swa + 1500),
        fontsize=9,
        fontweight="bold",
        color=COLORS["Flex SWA"],
        arrowprops=dict(arrowstyle="->", color=COLORS["Flex SWA"], lw=1.2),
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096])
    ax.set_xticklabels([256, 512, 1024, 2048, 4096])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Training Throughput: SDPA vs FlexAttention")
    ax.legend(loc="upper left")
    try:
        sns.despine(ax=ax)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "fig6_training_throughput", png_only)


# ── Figure 7: Combined Latency (exp1a + exp1b) ──────────────────────────────
def fig7_combined_latency(png_only=False):
    d_inf = load_json("exp1a_inference_latency_memory.json")
    d_trn = load_json("exp1b_training_latency.json")
    if not d_inf or not d_trn:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.8), sharey=False)

    configs = [
        ("SDPA causal", "SDPA causal", COLORS["SDPA causal"], "o"),
        ("SDPA sliding_window(256)", "SDPA SWA(256)", COLORS["SDPA SWA"], "o"),
        ("Flex causal", "Flex causal", COLORS["Flex causal"], "^"),
        ("Flex sliding_window(256)", "Flex SWA(256)", COLORS["Flex SWA"], "^"),
    ]

    # Left panel: inference
    b_inf = d_inf["benchmarks"]
    for key, label, color, marker in configs:
        seq, means, stds = extract_series_with_std(b_inf, key)
        means = np.array(means)
        ax1.plot(seq, means, marker=marker, color=color, linewidth=1.3, markersize=4, label=label)

    # Annotate inference SWA speedup
    sdpa_swa_inf = b_inf["SDPA sliding_window(256)"]["4096"]["mean_ms"]
    flex_swa_inf = b_inf["Flex sliding_window(256)"]["4096"]["mean_ms"]
    ax1.annotate(
        f"{sdpa_swa_inf / flex_swa_inf:.2f}x",
        xy=(4096, flex_swa_inf),
        xytext=(2600, flex_swa_inf - 12),
        fontsize=8, fontweight="bold", color=COLORS["Flex SWA"],
        arrowprops=dict(arrowstyle="->", color=COLORS["Flex SWA"], lw=1),
    )

    ax1.set_xscale("log", base=2)
    ax1.set_xticks([256, 512, 1024, 2048, 4096])
    ax1.set_xticklabels([256, 512, 1024, 2048, 4096], fontsize=7)
    ax1.set_xlabel("Sequence Length", fontsize=9)
    ax1.set_ylabel("Latency (ms)", fontsize=9)
    ax1.set_title("(a) Inference", fontsize=10)
    ax1.tick_params(axis='y', labelsize=7)

    # Right panel: training
    b_trn = d_trn["benchmarks"]
    for key, label, color, marker in configs:
        seq, vals = extract_series(b_trn, key)
        ax2.plot(seq, vals, marker=marker, color=color, linewidth=1.3, markersize=4, label=label)

    # Annotate training SWA speedup
    sdpa_swa_trn = b_trn["SDPA sliding_window(256)"]["4096"]["mean_ms"]
    flex_swa_trn = b_trn["Flex sliding_window(256)"]["4096"]["mean_ms"]
    ax2.annotate(
        f"{sdpa_swa_trn / flex_swa_trn:.2f}x",
        xy=(4096, flex_swa_trn),
        xytext=(2600, flex_swa_trn - 50),
        fontsize=8, fontweight="bold", color=COLORS["Flex SWA"],
        arrowprops=dict(arrowstyle="->", color=COLORS["Flex SWA"], lw=1),
    )

    ax2.set_xscale("log", base=2)
    ax2.set_xticks([256, 512, 1024, 2048, 4096])
    ax2.set_xticklabels([256, 512, 1024, 2048, 4096], fontsize=7)
    ax2.set_xlabel("Sequence Length", fontsize=9)
    ax2.set_ylabel("Latency (ms)", fontsize=9)
    ax2.set_title("(b) Training (Fwd + Bwd)", fontsize=10)
    ax2.tick_params(axis='y', labelsize=7)

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, 1.08))

    try:
        sns.despine(ax=ax1)
        sns.despine(ax=ax2)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "fig7_combined_latency", png_only)


# ── Supplementary Figures ────────────────────────────────────────────────────
def figS1_training_latency(png_only=False):
    d = load_json("exp1b_training_latency.json")
    if not d:
        return
    b = d["benchmarks"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    configs = [
        ("SDPA causal", "SDPA causal", COLORS["SDPA causal"], "o"),
        ("SDPA sliding_window(256)", "SDPA SWA(256)", COLORS["SDPA SWA"], "o"),
        ("Flex causal", "Flex causal", COLORS["Flex causal"], "^"),
        ("Flex sliding_window(256)", "Flex SWA(256)", COLORS["Flex SWA"], "^"),
    ]
    for key, label, color, marker in configs:
        seq, vals = extract_series(b, key)
        ax.plot(seq, vals, marker=marker, label=label, color=color, linewidth=1.5, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096])
    ax.set_xticklabels([256, 512, 1024, 2048, 4096])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Training Latency (Fwd + Bwd): SDPA vs FlexAttention")
    ax.legend(loc="upper left")
    try:
        sns.despine(ax=ax)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "figS1_training_latency", png_only)


def figS2_block_mask_ablation(png_only=False):
    d = load_json("exp3_block_mask_ablation.json")
    if not d:
        return
    b = d["benchmarks"]

    seq_lens = ["256", "512", "1024", "2048", "4096"]
    pairs = [
        ("Causal + block_mask", "Causal (no block_mask)", "Causal"),
        ("Sliding_window + block_mask", "Sliding_window (no block_mask)", "SWA(256)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (with_key, without_key, title) in zip(axes, pairs):
        x = np.arange(len(seq_lens))
        width = 0.35
        vals_with = [b[with_key][s]["mean_ms"] for s in seq_lens]
        vals_without = [b[without_key][s]["mean_ms"] for s in seq_lens]
        ax.bar(x - width / 2, vals_with, width, label="With block_mask", color=COLORS["Flex causal"])
        ax.bar(x + width / 2, vals_without, width, label="Without block_mask", color="#999999")
        ax.set_xticks(x)
        ax.set_xticklabels([int(s) for s in seq_lens], fontsize=8)
        ax.set_xlabel("Sequence Length")
        ax.set_title(f"{title}: Block Mask Ablation")
        ax.legend(fontsize=8)

        # Annotate speedup at 4096
        ratio = vals_without[-1] / vals_with[-1]
        ax.text(x[-1], vals_with[-1] + 2, f"{ratio:.2f}x", ha="center", fontsize=8, fontweight="bold", color=COLORS["Flex causal"])

    axes[0].set_ylabel("Latency (ms)")
    try:
        for ax in axes:
            sns.despine(ax=ax)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "figS2_block_mask_ablation", png_only)


def figS3_mistral_swa(png_only=False):
    d = load_json("exp4_mistral_swa.json")
    if not d:
        return
    b = d["benchmarks"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    configs = [
        ("Native Mistral SWA (SDPA)", "Native SDPA", COLORS["SDPA causal"], "o"),
        ("Flex Mistral SWA", "FlexAttention", COLORS["Flex causal"], "^"),
    ]
    for key, label, color, marker in configs:
        seq, vals = extract_series(b, key)
        ax.plot(seq, vals, marker=marker, label=label, color=color, linewidth=1.5, markersize=5)

    sdpa_4k = b["Native Mistral SWA (SDPA)"]["4096"]["mean_ms"]
    flex_4k = b["Flex Mistral SWA"]["4096"]["mean_ms"]
    ax.annotate(
        f"{sdpa_4k / flex_4k:.2f}x",
        xy=(4096, flex_4k),
        xytext=(3200, flex_4k - 12),
        fontsize=9,
        fontweight="bold",
        color=COLORS["Flex causal"],
        arrowprops=dict(arrowstyle="->", color=COLORS["Flex causal"], lw=1.2),
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096])
    ax.set_xticklabels([256, 512, 1024, 2048, 4096])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Mistral SWA: Native SDPA vs FlexAttention")
    ax.legend(loc="upper left")
    try:
        sns.despine(ax=ax)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "figS3_mistral_swa", png_only)


def figS4_oom_limits(png_only=False):
    d = load_json("exp5_oom_limit.json")
    if not d:
        return
    lim = d["limits"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, mode in zip(axes, ["inference", "training"]):
        methods = list(lim[mode].keys())
        max_seqs = [lim[mode][m]["max_seq_len"] for m in methods]
        peak_mems = [lim[mode][m]["peak_mem_mb"] / 1024 for m in methods]
        # Short labels
        labels = [m.replace("sliding_window(256)", "SWA(256)") for m in methods]
        colors = [COLORS["Flex causal"] if "Flex" in m else COLORS["SDPA causal"] for m in methods]

        y = np.arange(len(methods))
        ax.barh(y, max_seqs, color=colors, edgecolor="white")
        for i, (seq, mem) in enumerate(zip(max_seqs, peak_mems)):
            ax.text(seq + 500, i, f"{seq:,} ({mem:.1f} GB)", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Max Sequence Length")
        ax.set_title(f"OOM Limits ({mode.title()})")
        ax.set_xscale("log", base=2)

    try:
        for ax in axes:
            sns.despine(ax=ax)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "figS4_oom_limits", png_only)


def figS5_compound_masks(png_only=False):
    d = load_json("exp6_compound_masks.json")
    if not d:
        return
    b = d["benchmarks"]

    seq_lens = ["256", "1024", "4096"]
    methods = [
        ("Flex Causal", "Causal", COLORS["SDPA causal"]),
        ("Flex Sliding Window", "SWA(256)", COLORS["Flex causal"]),
        ("Flex ALiBi + SWA", "ALiBi+SWA(256)", COLORS["Flex SWA"]),
    ]

    x = np.arange(len(seq_lens))
    width = 0.25
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, (key, label, color) in enumerate(methods):
        vals = [b[key][s]["mean_ms"] for s in seq_lens]
        ax.bar(x + i * width, vals, width, label=label, color=color, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels([int(s) for s in seq_lens])
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Training Latency (ms)")
    ax.set_title("Compound Masks: ALiBi + SWA Composition")
    ax.legend(loc="upper left")
    try:
        sns.despine(ax=ax)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "figS5_compound_masks", png_only)


def figS6_batch_sensitivity(png_only=False):
    d = load_json("exp9_batch_sensitivity.json")
    if not d:
        return
    b = d["benchmarks"]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    configs = [
        ("SDPA causal", "SDPA causal", COLORS["SDPA causal"], "o"),
        ("Flex causal", "Flex causal", COLORS["Flex causal"], "^"),
    ]
    for key, label, color, marker in configs:
        batch_sizes = sorted(b[key].keys(), key=int)
        vals = [b[key][bs]["mean_ms"] for bs in batch_sizes]
        ax.plot([int(bs) for bs in batch_sizes], vals, marker=marker, label=label, color=color, linewidth=1.5, markersize=5)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Batch Sensitivity: SDPA vs FlexAttention (seq=1024)")
    ax.legend(loc="upper left")
    try:
        sns.despine(ax=ax)
    except Exception:
        pass
    fig.tight_layout()
    save_fig(fig, "figS6_batch_sensitivity", png_only)


# ── Main ─────────────────────────────────────────────────────────────────────
MAIN_FIGURES = {
    1: ("Inference Latency (exp1a)", fig1_inference_latency),
    2: ("Training Equivalence (exp2)", fig2_training_equivalence),
    3: ("Document Masking (exp8)", fig3_document_masking),
    4: ("GQA Heatmap (exp12)", fig4_gqa_heatmap),
    5: ("Decode Caveat (exp10)", fig5_decode_caveat),
    6: ("Training Throughput (exp11)", fig6_training_throughput),
    7: ("Combined Latency (exp1a+1b)", fig7_combined_latency),
}

SUPP_FIGURES = {
    "S1": ("Training Latency (exp1b)", figS1_training_latency),
    "S2": ("Block Mask Ablation (exp3)", figS2_block_mask_ablation),
    "S3": ("Mistral SWA (exp4)", figS3_mistral_swa),
    "S4": ("OOM Limits (exp5)", figS4_oom_limits),
    "S5": ("Compound Masks (exp6)", figS5_compound_masks),
    "S6": ("Batch Sensitivity (exp9)", figS6_batch_sensitivity),
}


def main():
    parser = argparse.ArgumentParser(description="Generate FlexAttention experiment figures")
    parser.add_argument("--all", action="store_true", help="Generate supplementary figures too")
    parser.add_argument("--fig", nargs="+", help="Generate specific figures (e.g., 1 4 S2)")
    parser.add_argument("--png-only", action="store_true", help="Skip PDF output")
    args = parser.parse_args()

    print("FlexAttention Results Plotter")
    print(f"Output: {FIGURES_DIR}/\n")

    if args.fig:
        for f in args.fig:
            if f.startswith("S"):
                if f in SUPP_FIGURES:
                    name, fn = SUPP_FIGURES[f]
                    print(f"Fig {f}: {name}")
                    fn(args.png_only)
            else:
                idx = int(f)
                if idx in MAIN_FIGURES:
                    name, fn = MAIN_FIGURES[idx]
                    print(f"Fig {idx}: {name}")
                    fn(args.png_only)
    else:
        for idx, (name, fn) in MAIN_FIGURES.items():
            print(f"Fig {idx}: {name}")
            fn(args.png_only)

        if args.all:
            print()
            for idx, (name, fn) in SUPP_FIGURES.items():
                print(f"Fig {idx}: {name}")
                fn(args.png_only)

    print("\nDone!")


if __name__ == "__main__":
    main()
