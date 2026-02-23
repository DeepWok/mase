import json
import logging
import os
from pathlib import Path

from safetensors.torch import save_file, load_file


def save_layer_checkpoint(model, layer_idx, checkpoint_dir, model_name="quantized_model"):
    if checkpoint_dir is None:
        return

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    layer_checkpoint_file = checkpoint_path / f"{model_name}_layer_{layer_idx}.safetensors"

    if layer_checkpoint_file.exists() and layer_checkpoint_file.is_dir():
        import shutil
        shutil.rmtree(layer_checkpoint_file)
        logging.info(f"Removed existing directory: {layer_checkpoint_file}")

    logging.info(f"Saving layer {layer_idx} checkpoint to {layer_checkpoint_file}")

    try:
        layer_state_dict = {}
        layer_prefix = f"model.layers.{layer_idx}."

        for name, param in model.named_parameters():
            if name.startswith(layer_prefix):
                relative_name = name[len(layer_prefix):]
                layer_state_dict[relative_name] = param.detach().cpu()

        if not layer_state_dict:
            logging.warning(f"No parameters found for layer {layer_idx} with prefix {layer_prefix}")
            return

        save_file(layer_state_dict, str(layer_checkpoint_file))

        metadata = {
            "layer_idx": layer_idx,
            "total_layers": len(model.model.layers),
            "checkpoint_file": str(layer_checkpoint_file),
            "model_name": model_name,
            "num_parameters": len(layer_state_dict),
            "parameter_names": list(layer_state_dict.keys())
        }

        metadata_file = checkpoint_path / f"{model_name}_layer_{layer_idx}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logging.info(f"Layer {layer_idx} checkpoint saved successfully ({len(layer_state_dict)} parameters)")

    except Exception as e:
        logging.error(f"Failed to save layer {layer_idx} checkpoint: {e}")


def detect_quantized_layers(checkpoint_dir, model_name="quantized_model"):
    if checkpoint_dir is None or not os.path.exists(checkpoint_dir):
        return {}

    checkpoint_path = Path(checkpoint_dir)
    checkpoints = list(checkpoint_path.glob(f"{model_name}_layer_*.safetensors"))

    quantized_layers = {}
    for checkpoint in checkpoints:
        try:
            layer_idx = int(checkpoint.stem.split('_layer_')[-1])
            quantized_layers[layer_idx] = str(checkpoint)
        except ValueError:
            continue

    return quantized_layers


def load_layer_checkpoint(model, layer_idx, checkpoint_file):
    if not os.path.exists(checkpoint_file):
        logging.error(f"Layer checkpoint file {checkpoint_file} not found")
        return False

    try:
        layer_state_dict = load_file(checkpoint_file)

        layer_prefix = f"model.layers.{layer_idx}."
        model_state_dict = {}

        for param_name, param_value in layer_state_dict.items():
            full_param_name = layer_prefix + param_name
            model_state_dict[full_param_name] = param_value

        model.load_state_dict(model_state_dict, strict=False)

        logging.info(f"Successfully loaded layer {layer_idx} from checkpoint")
        return True

    except Exception as e:
        logging.error(f"Failed to load layer {layer_idx} checkpoint: {e}")
        return False


def auto_load_quantized_layers(model, checkpoint_dir, model_name="quantized_model"):
    quantized_layers = detect_quantized_layers(checkpoint_dir, model_name)

    if not quantized_layers:
        logging.info("No quantized layer checkpoints found")
        return -1

    loaded_count = 0
    max_layer_idx = -1

    for layer_idx in sorted(quantized_layers.keys()):
        checkpoint_file = quantized_layers[layer_idx]
        if load_layer_checkpoint(model, layer_idx, checkpoint_file):
            loaded_count += 1
            max_layer_idx = layer_idx
        else:
            logging.warning(f"Failed to load layer {layer_idx}, stopping auto-load")
            break

    logging.info(f"Auto-loaded {loaded_count} quantized layers (up to layer {max_layer_idx})")
    return max_layer_idx
