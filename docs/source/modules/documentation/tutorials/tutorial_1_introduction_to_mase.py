# Tutorial 1: Introduction to the Mase IR, MaseGraph and Torch FX passes

import os
import platform
from pathlib import Path

# ── macOS: ensure Graphviz is on PATH ──────────────────────────────────────────
if platform.system() == "Darwin":
    homebrew_bin = "/opt/homebrew/bin"
    if not os.path.exists(homebrew_bin):
        homebrew_bin = "/usr/local/bin"
    if os.path.exists(homebrew_bin):
        os.environ["PATH"] = homebrew_bin + ":" + os.environ.get("PATH", "")

print("=" * 60, flush=True)
print("Tutorial 1: Introduction to MaseGraph & FX passes", flush=True)
print("=" * 60, flush=True)

# ── Step 1: Load pretrained model ─────────────────────────────────────────────
print("\n[1/6] Loading pretrained bert-tiny from HuggingFace...", flush=True)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
print(f"      Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

# ── Step 2: Generate FX / MaseGraph ───────────────────────────────────────────
print("\n[2/6] Building MaseGraph and drawing SVG...", flush=True)
from chop import MaseGraph

mg = MaseGraph(model)
mg.draw("bert-base-uncased.svg")
print("      Graph saved to bert-base-uncased.svg", flush=True)

# FX node type sanity check
import torch
random_tensor = torch.randn(2, 2)
assert torch.equal(torch.relu(random_tensor), random_tensor.relu())
assert torch.equal(torch.relu(random_tensor), torch.nn.ReLU()(random_tensor))
print("      FX node type sanity check passed.", flush=True)

# ── Step 3: Raise to Mase IR ──────────────────────────────────────────────────
print("\n[3/6] Running metadata analysis passes...", flush=True)
import chop.passes as passes

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dummy_input = tokenizer(
    [
        "AI may take over the world one day",
        "This is why you should learn ADLS",
    ],
    return_tensors="pt",
)

mg, _ = passes.init_metadata_analysis_pass(mg)
print("      init_metadata_analysis_pass  ✓", flush=True)

mg, _ = passes.add_common_metadata_analysis_pass(
    mg,
    pass_args={"dummy_in": dummy_input, "add_value": False},
)
print("      add_common_metadata_analysis_pass  ✓", flush=True)

# ── Step 4: Analysis pass – count dropout ─────────────────────────────────────
print("\n[4/6] Running count_dropout_analysis_pass...", flush=True)
from chop.tools import get_logger

logger = get_logger("mase_logger")
logger.setLevel("INFO")


def count_dropout_analysis_pass(mg, pass_args={}):
    dropout_modules = 0
    for node in mg.fx_graph.nodes:
        if node.op == "call_module" and "dropout" in node.target:
            logger.info(f"  Found dropout module: {node.target}")
            dropout_modules += 1
    return mg, {"dropout_count": dropout_modules}


mg, pass_out = count_dropout_analysis_pass(mg)
print(f"      Dropout count: {pass_out['dropout_count']}", flush=True)

# ── Step 5: Transform pass – remove dropout ───────────────────────────────────
print("\n[5/6] Running remove_dropout_transform_pass...", flush=True)


def remove_dropout_transform_pass(mg, pass_args={}):
    removed = 0
    for node in list(mg.fx_graph.nodes):
        if node.op == "call_module" and "dropout" in node.target:
            parent_node = node.args[0]
            node.replace_all_uses_with(parent_node)
            mg.fx_graph.erase_node(node)
            removed += 1
    return mg, {"removed": removed}


mg, info = remove_dropout_transform_pass(mg)
print(f"      Removed {info['removed']} dropout node(s).", flush=True)

mg, pass_out = count_dropout_analysis_pass(mg)
assert pass_out["dropout_count"] == 0
print("      Verified: 0 dropout nodes remain  ✓", flush=True)

# ── Step 6: Export / reload checkpoint ────────────────────────────────────────
print("\n[6/6] Exporting MaseGraph checkpoint...", flush=True)
mg.export(f"{Path.home()}/tutorial_1")
print(f"      Exported to {Path.home()}/tutorial_1", flush=True)

new_mg = MaseGraph.from_checkpoint(f"{Path.home()}/tutorial_1")
print("      Reloaded from checkpoint  ✓", flush=True)

print("\n" + "=" * 60, flush=True)
print("Tutorial 1 complete!", flush=True)
print("=" * 60, flush=True)
