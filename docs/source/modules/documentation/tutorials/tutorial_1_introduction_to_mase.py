"""Tutorial 1 script entrypoint.

Run from repository root:
    uv run python docs/source/modules/documentation/tutorials/tutorial_1_introduction_to_mase.py
"""

import os
import platform
from pathlib import Path

import torch
import chop.passes as passes
from chop import MaseGraph
from chop.tools import get_logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def ensure_graphviz_path_on_macos() -> None:
    """Ensure Graphviz binaries are visible on macOS."""
    if platform.system() != "Darwin":
        return
    homebrew_bin = "/opt/homebrew/bin"
    if not os.path.exists(homebrew_bin):
        homebrew_bin = "/usr/local/bin"
    if os.path.exists(homebrew_bin):
        os.environ["PATH"] = homebrew_bin + ":" + os.environ.get("PATH", "")


def count_dropout_analysis_pass(mg, logger, pass_args={}):
    """Count dropout modules in a graph."""
    del pass_args
    dropout_modules = 0
    for node in mg.fx_graph.nodes:
        if node.op == "call_module" and "dropout" in node.target:
            logger.info("Found dropout module: %s", node.target)
            dropout_modules += 1
    return mg, {"dropout_count": dropout_modules}


def remove_dropout_transform_pass(mg, logger, pass_args={}):
    """Remove dropout nodes from a graph."""
    del pass_args
    for node in list(mg.fx_graph.nodes):
        if node.op == "call_module" and "dropout" in node.target:
            logger.info("Removing dropout module: %s", node.target)
            parent_node = node.args[0]
            node.replace_all_uses_with(parent_node)
            mg.fx_graph.erase_node(node)
    return mg, {}


def main() -> None:
    ensure_graphviz_path_on_macos()

    print("=" * 60, flush=True)
    print("Tutorial 1: Introduction to MaseGraph & FX passes", flush=True)
    print("=" * 60, flush=True)

    # [load_model:start]
    print("\n[1/6] Loading pretrained bert-tiny from HuggingFace...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")
    print(
        f"      Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    # [load_model:end]

    # [build_graph:start]
    print("\n[2/6] Building MaseGraph and drawing SVG...", flush=True)
    mg = MaseGraph(model)
    script_dir = Path(__file__).resolve().parent
    graph_path = script_dir / "bert-base-uncased.svg"
    mg.draw(str(graph_path))
    print(f"      Graph saved to {graph_path}", flush=True)

    random_tensor = torch.randn(2, 2)
    assert torch.equal(torch.relu(random_tensor), random_tensor.relu())
    assert torch.equal(torch.relu(random_tensor), torch.nn.ReLU()(random_tensor))
    print("      FX node type sanity check passed.", flush=True)
    # [build_graph:end]

    # [raise_ir:start]
    print("\n[3/6] Running metadata analysis passes...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dummy_input = tokenizer(
        [
            "AI may take over the world one day",
            "This is why you should learn ADLS",
        ],
        return_tensors="pt",
    )
    mg, _ = passes.init_metadata_analysis_pass(mg)
    mg, _ = passes.add_common_metadata_analysis_pass(
        mg,
        pass_args={"dummy_in": dummy_input, "add_value": False},
    )
    print("      Metadata analysis passes completed  ✓", flush=True)
    # [raise_ir:end]

    # [analysis_pass:start]
    print("\n[4/6] Running count_dropout_analysis_pass...", flush=True)
    logger = get_logger("mase_logger")
    logger.setLevel("INFO")
    mg, pass_out = count_dropout_analysis_pass(mg, logger)
    print(f"      Dropout count: {pass_out['dropout_count']}", flush=True)
    # [analysis_pass:end]

    # [transform_pass:start]
    print("\n[5/6] Running remove_dropout_transform_pass...", flush=True)
    mg, _ = remove_dropout_transform_pass(mg, logger)
    mg, pass_out = count_dropout_analysis_pass(mg, logger)
    assert pass_out["dropout_count"] == 0
    print("      Verified: 0 dropout nodes remain  ✓", flush=True)
    # [transform_pass:end]

    # [export:start]
    print("\n[6/6] Exporting MaseGraph checkpoint...", flush=True)
    export_dir = f"{Path.home()}/tutorial_1"
    mg.export(export_dir)
    print(f"      Exported to {export_dir}", flush=True)
    _ = MaseGraph.from_checkpoint(export_dir)
    print("      Reloaded from checkpoint  ✓", flush=True)
    # [export:end]

    print("\n" + "=" * 60, flush=True)
    print("Tutorial 1 complete!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
