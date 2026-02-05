#!/usr/bin/env python3
"""
Extract and analyze best model configuration from mixed precision search.

Usage:
    python extract_best_config.py --model_path mixed_precision_best_tpe.pt
    python extract_best_config.py --model_path mixed_precision_best_tpe.pt --output_path analysis.txt
"""

import argparse
from pathlib import Path
from datetime import datetime
import torch


def reconstruct_layer_configs(best_params: dict) -> dict:
    """
    Reconstruct layer_configs from Optuna's best_params.
    
    best_params has keys like:
        'bert.encoder.layer.0.attention.self.query_type': <class 'torch.nn.Linear'>
        'bert.encoder.layer.0.attention.self.query_width': 16
        'bert.encoder.layer.0.attention.self.query_frac_width': 4
    """
    layer_configs = {}
    
    # Find all layer names by looking for _type keys
    layer_names = set()
    for key in best_params.keys():
        if key.endswith("_type"):
            layer_name = key[:-5]  # Remove '_type' suffix
            layer_names.add(layer_name)
    
    for layer_name in sorted(layer_names):
        layer_type = best_params.get(f"{layer_name}_type")
        
        # Check if it's quantized (LinearInteger) or full precision (Linear)
        type_str = str(layer_type)
        
        if "LinearInteger" in type_str:
            width = best_params.get(f"{layer_name}_width")
            frac_width = best_params.get(f"{layer_name}_frac_width")
            layer_configs[layer_name] = {
                "type": "LinearInteger",
                "width": width,
                "frac_width": frac_width
            }
        else:
            # Full precision nn.Linear
            layer_configs[layer_name] = {"type": "Linear"}
    
    return layer_configs


def extract_config(model_path: str, output_path: str | None = None) -> None:
    """Extract configuration from best model .pt file and save analysis."""
    
    # Load the payload (weights_only=False needed for custom objects like layer_configs)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Debug: show what keys are in the payload
    print(f"Keys in payload: {list(payload.keys())}")
    
    # Determine output path
    if output_path is None:
        model_p = Path(model_path)
        output_path = model_p.parent / f"{model_p.stem}_analysis.txt"
    else:
        output_p = Path(output_path)
        if output_p.is_dir():
            # If output_path is a directory, create filename based on model name
            model_p = Path(model_path)
            output_path = output_p / f"{model_p.stem}_analysis.txt"
    
    # Extract information
    best_accuracy = payload.get("best_accuracy", "N/A")
    best_params = payload.get("best_params", {})
    layer_configs = payload.get("layer_configs", {})
    checkpoint = payload.get("checkpoint", "N/A")
    saved_at = payload.get("saved_at", "N/A")
    
    # If layer_configs is empty, try to reconstruct from best_params
    if not layer_configs and best_params:
        print("\nNo layer_configs found. Reconstructing from best_params...")
        layer_configs = reconstruct_layer_configs(best_params)
    
    if not layer_configs:
        print("\nWARNING: No layer configuration found in payload.")
        print("Available keys:", list(payload.keys()))
        if best_params:
            print("\nBest params available:")
            for k, v in sorted(best_params.items())[:20]:  # Show first 20
                print(f"  {k}: {v}")
            if len(best_params) > 20:
                print(f"  ... and {len(best_params) - 20} more")
        return
    
    # Analyze layer configurations
    quantized_layers = []
    full_precision_layers = []
    
    for name, cfg in layer_configs.items():
        if cfg["type"] == "LinearInteger":
            quantized_layers.append({
                "name": name,
                "width": cfg["width"],
                "frac_width": cfg["frac_width"]
            })
        else:
            full_precision_layers.append(name)
    
    # Compute statistics
    total_layers = len(layer_configs)
    num_quantized = len(quantized_layers)
    num_full_precision = len(full_precision_layers)
    
    # Width distribution
    width_counts = {}
    frac_width_counts = {}
    for layer in quantized_layers:
        w = layer["width"]
        fw = layer["frac_width"]
        width_counts[w] = width_counts.get(w, 0) + 1
        frac_width_counts[fw] = frac_width_counts.get(fw, 0) + 1
    
    # Build output text
    lines = []
    lines.append("=" * 70)
    lines.append("MIXED PRECISION SEARCH - BEST MODEL ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Analysis generated: {datetime.now()}")
    lines.append(f"Model file: {model_path}")
    lines.append(f"Model saved at: {saved_at}")
    lines.append(f"Base checkpoint: {checkpoint}")
    lines.append("")
    
    # Summary
    lines.append("-" * 70)
    lines.append("SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Best Accuracy: {best_accuracy:.4f}" if isinstance(best_accuracy, float) else f"Best Accuracy: {best_accuracy}")
    lines.append(f"Total Linear Layers: {total_layers}")
    lines.append(f"Quantized Layers: {num_quantized} ({100*num_quantized/total_layers:.1f}%)")
    lines.append(f"Full Precision Layers: {num_full_precision} ({100*num_full_precision/total_layers:.1f}%)")
    lines.append("")
    
    # Width distribution
    lines.append("-" * 70)
    lines.append("QUANTIZATION DISTRIBUTION")
    lines.append("-" * 70)
    lines.append("Width distribution:")
    for w in sorted(width_counts.keys()):
        count = width_counts[w]
        pct = 100 * count / num_quantized if num_quantized > 0 else 0
        lines.append(f"  width={w}: {count} layers ({pct:.1f}%)")
    
    lines.append("")
    lines.append("Fractional width distribution:")
    for fw in sorted(frac_width_counts.keys()):
        count = frac_width_counts[fw]
        pct = 100 * count / num_quantized if num_quantized > 0 else 0
        lines.append(f"  frac_width={fw}: {count} layers ({pct:.1f}%)")
    lines.append("")
    
    # Per-layer details - Quantized
    lines.append("-" * 70)
    lines.append("QUANTIZED LAYERS (LinearInteger)")
    lines.append("-" * 70)
    lines.append(f"{'Layer Name':<55} {'Width':<8} {'Frac':<8}")
    lines.append("-" * 70)
    for layer in quantized_layers:
        name = layer["name"]
        # Truncate long names
        if len(name) > 54:
            name = "..." + name[-51:]
        lines.append(f"{name:<55} {layer['width']:<8} {layer['frac_width']:<8}")
    lines.append("")
    
    # Per-layer details - Full precision
    lines.append("-" * 70)
    lines.append("FULL PRECISION LAYERS (nn.Linear)")
    lines.append("-" * 70)
    for name in full_precision_layers:
        if len(name) > 68:
            name = "..." + name[-65:]
        lines.append(f"  {name}")
    lines.append("")
    
    # Configuration combinations
    lines.append("-" * 70)
    lines.append("UNIQUE CONFIGURATIONS")
    lines.append("-" * 70)
    config_combos = {}
    for layer in quantized_layers:
        key = (layer["width"], layer["frac_width"])
        config_combos[key] = config_combos.get(key, 0) + 1
    
    lines.append(f"{'(width, frac_width)':<25} {'Count':<10} {'Percentage':<10}")
    lines.append("-" * 45)
    for (w, fw), count in sorted(config_combos.items()):
        pct = 100 * count / num_quantized if num_quantized > 0 else 0
        lines.append(f"({w}, {fw}){'':<17} {count:<10} {pct:.1f}%")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("END OF ANALYSIS")
    lines.append("=" * 70)
    
    # Write to file
    output_text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(output_text)
    
    # Also print to console
    print(output_text)
    print(f"\nAnalysis saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and analyze best model configuration"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the best model .pt file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for analysis text file (default: <model_name>_analysis.txt)"
    )
    args = parser.parse_args()
    
    extract_config(args.model_path, args.output_path)


if __name__ == "__main__":
    main()