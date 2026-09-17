"""
Main GPTQ orchestration: run_gptq(network, gptq_config).

Ported from Coprocessor_for_Llama/acc_simulator/gptq/quant.py,
adapted to use Mase config dicts and write quantized weights
back in-place to nn.Linear modules (no module replacement here).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gptq import GPTQ
from .quantize_dispatch import quantize_tensor
from .utils import find_qlayers, cleanup_memory
from .data_utils import get_loaders
from .checkpoint import save_layer_checkpoint, auto_load_quantized_layers
from chop.nn.quantized.modules.phase_config import (
    DECODE_FP_EXPERT_DOWN_ATTR,
    DECODE_FP_EXPERT_GATE_UP_ATTR,
    DECODE_FP_BIAS_ATTR,
    DECODE_FP_WEIGHT_ATTR,
    GPTQ_DECODE_EXPERT_DOWN_ATTR,
    GPTQ_DECODE_EXPERT_GATE_UP_ATTR,
    GPTQ_DECODE_WEIGHT_ATTR,
)

_VALID_GPTQ_PHASES = ("both", "decode", "prefill")
_DENSE_SEQUENTIAL = (
    ("self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"),
    ("self_attn.o_proj",),
    ("mlp.up_proj", "mlp.gate_proj"),
    ("mlp.down_proj",),
)


class _WeightView:
    """Minimal GPTQ adapter for one expert's fused weight slice."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight


class _CalibrationCaptureComplete(RuntimeError):
    pass


def _module_device(module: nn.Module, module_name: str) -> torch.device:
    devices = {param.device for param in module.parameters(recurse=True)}
    devices.update(buffer.device for buffer in module.buffers(recurse=True))
    devices = {device for device in devices if device.type != "meta"}
    if not devices:
        return torch.device("cpu")
    if len(devices) != 1:
        formatted = ", ".join(sorted(str(device) for device in devices))
        raise RuntimeError(
            "GPTQ device_map_aware requires layer-level sharding, but "
            f"{module_name} spans multiple devices: {formatted}"
        )
    return next(iter(devices))


def _move_to_device(value, device: torch.device):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _run_layer_sample(
    *,
    layer,
    sample,
    outs,
    sample_idx,
    rope,
    attention_mask,
    position_ids,
    layer_device,
):
    x = sample.unsqueeze(0).to(layer_device)
    local_mask = _move_to_device(attention_mask, layer_device)
    local_positions = _move_to_device(position_ids, layer_device)
    local_rope = rope.to(layer_device)
    cos, sin = local_rope(x, local_positions)
    output = layer(
        x,
        attention_mask=local_mask,
        position_embeddings=(cos, sin),
    )
    if isinstance(output, (tuple, list)):
        output = output[0]
    outs[sample_idx].copy_(
        output.squeeze(0).to(device=outs.device, dtype=outs.dtype)
    )


def _snapshot_fp_linear_weights(layers) -> None:
    """Snapshot FP decoder weights before GPTQ mutates them.

    Snapshots live on CPU (``_mase_fp_weight``) so large models don't hold a
    second GPU copy. Must run before checkpoint resume, which loads already
    GPTQ-quantised weights into the network.
    """

    for layer in layers:
        for module in layer.modules():
            if isinstance(module, nn.Linear) and not hasattr(
                module, "_mase_fp_weight"
            ):
                module._mase_fp_weight = module.weight.detach().clone().cpu()
            if (
                hasattr(module, "gate_up_proj")
                and isinstance(module.gate_up_proj, nn.Parameter)
                and module.gate_up_proj.ndim == 3
                and hasattr(module, "down_proj")
                and isinstance(module.down_proj, nn.Parameter)
            ):
                if not hasattr(module, "_mase_fp_gate_up_proj"):
                    module._mase_fp_gate_up_proj = (
                        module.gate_up_proj.detach().clone().cpu()
                    )
                    module._mase_fp_down_proj = (
                        module.down_proj.detach().clone().cpu()
                    )


def _finalize_gptq_phase(network, phase: str) -> None:
    """Re-home GPTQ results according to the target phase.

    - ``both`` (legacy): weights stay mutated in place; both phases see them.
    - ``decode``: the GPTQ result is stashed as ``_mase_gptq_weight_decode``
      (picked up as the decode weight bank at module replacement) and the FP
      snapshot is restored into ``weight`` so prefill stays unquantised —
      the decode-side disaggregated-serving flow.
    - ``prefill``: weights stay mutated in place (prefill bank) and the FP
      snapshot is stashed as ``_mase_decode_weight_fp`` for an ``fp_only``
      decode policy — the prefill-side flow.
    """

    if phase == "both":
        return
    for layer in network.model.layers:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                fp_weight = getattr(module, "_mase_fp_weight", None)
                if fp_weight is None:
                    continue
                if phase == "decode":
                    setattr(
                        module,
                        GPTQ_DECODE_WEIGHT_ATTR,
                        module.weight.detach().clone().cpu(),
                    )
                    module.weight.data.copy_(
                        fp_weight.to(
                            device=module.weight.device, dtype=module.weight.dtype
                        )
                    )
                else:  # prefill
                    setattr(module, DECODE_FP_WEIGHT_ATTR, fp_weight)
                    if module.bias is not None and not hasattr(
                        module, DECODE_FP_BIAS_ATTR
                    ):
                        setattr(
                            module,
                            DECODE_FP_BIAS_ATTR,
                            module.bias.detach().clone().cpu(),
                        )
                del module._mase_fp_weight
                continue

            fp_gate_up = getattr(module, "_mase_fp_gate_up_proj", None)
            fp_down = getattr(module, "_mase_fp_down_proj", None)
            if fp_gate_up is None and fp_down is None:
                continue
            if fp_gate_up is None or fp_down is None:
                raise RuntimeError("incomplete Qwen3-MoE FP expert snapshot")
            if phase == "decode":
                setattr(
                    module,
                    GPTQ_DECODE_EXPERT_GATE_UP_ATTR,
                    module.gate_up_proj.detach().clone().cpu(),
                )
                setattr(
                    module,
                    GPTQ_DECODE_EXPERT_DOWN_ATTR,
                    module.down_proj.detach().clone().cpu(),
                )
                module.gate_up_proj.data.copy_(
                    fp_gate_up.to(
                        device=module.gate_up_proj.device,
                        dtype=module.gate_up_proj.dtype,
                    )
                )
                module.down_proj.data.copy_(
                    fp_down.to(
                        device=module.down_proj.device,
                        dtype=module.down_proj.dtype,
                    )
                )
            else:
                setattr(module, DECODE_FP_EXPERT_GATE_UP_ATTR, fp_gate_up)
                setattr(module, DECODE_FP_EXPERT_DOWN_ATTR, fp_down)
            del module._mase_fp_gate_up_proj
            del module._mase_fp_down_proj


def _get_qwen3_moe_experts(layer):
    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    if experts is None:
        return None
    gate_up = getattr(experts, "gate_up_proj", None)
    down = getattr(experts, "down_proj", None)
    if not isinstance(gate_up, nn.Parameter) or not isinstance(down, nn.Parameter):
        return None
    if gate_up.ndim != 3 or down.ndim != 3:
        return None
    return experts


def _collect_expert_inputs(
    *,
    layer,
    experts,
    inps,
    outs,
    nsamples,
    rope,
    attention_mask,
    position_ids,
    layer_device,
    target,
):
    if target not in {"gate_up", "down"}:
        raise ValueError(f"unknown expert GPTQ target {target!r}")
    chunks = [[] for _ in range(experts.num_experts)]
    hits = [0 for _ in range(experts.num_experts)]

    def hook(_, inputs, _output):
        hidden_states, top_k_index, _top_k_weights = inputs
        with torch.no_grad():
            expert_mask = F.one_hot(
                top_k_index, num_classes=experts.num_experts
            ).permute(2, 1, 0)
            for hit in torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero():
                expert_idx = int(hit[0].item())
                _, token_idx = torch.where(expert_mask[expert_idx])
                current = hidden_states[token_idx]
                if current.numel() == 0:
                    continue
                if target == "down":
                    gate, up = F.linear(
                        current, experts.gate_up_proj[expert_idx]
                    ).chunk(2, dim=-1)
                    current = experts.act_fn(gate) * up
                chunks[expert_idx].append(current.detach().cpu())
                hits[expert_idx] += int(current.shape[0])

    handle = experts.register_forward_hook(hook)
    try:
        for sample_idx in range(nsamples):
            _run_layer_sample(
                layer=layer,
                sample=inps[sample_idx],
                outs=outs,
                sample_idx=sample_idx,
                rope=rope,
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_device=layer_device,
            )
    finally:
        handle.remove()
    return chunks, hits


def _quantize_expert_slice(
    *,
    weight,
    chunks,
    fmt,
    weight_config,
    quantile_search,
    clip_search_y,
    cali_batch_size,
    layer_name,
):
    if not chunks:
        return quantize_tensor(
            weight.data,
            block_dim=1,
            fmt=fmt,
            config=weight_config,
            quantile_search=quantile_search,
        )
    gptq = GPTQ(_WeightView(weight))
    for chunk in chunks:
        gptq.add_batch(chunk.to(weight.device).unsqueeze(0).data, None)
    activation = None
    if clip_search_y:
        activation = torch.cat(chunks, dim=0).unsqueeze(0).to(weight.device)
    quantized = gptq.fasterquant(
        activation=activation,
        fmt=fmt,
        weight_config=weight_config,
        percdamp=0.01,
        cali_batch_size=cali_batch_size,
        layer_name=layer_name,
        quant_search=quantile_search,
    )
    if quantized.shape != weight.shape:
        raise RuntimeError(
            f"GPTQ returned {tuple(quantized.shape)} for {tuple(weight.shape)}"
        )
    gptq.free()
    return quantized


def _quantize_qwen3_moe_experts(
    *,
    layer,
    layer_idx,
    inps,
    outs,
    nsamples,
    rope,
    attention_mask,
    position_ids,
    fmt,
    weight_config,
    quantile_search,
    clip_search_y,
    cali_batch_size,
    layer_device,
    min_expert_calibration_hits,
):
    experts = _get_qwen3_moe_experts(layer)
    if experts is None:
        return None

    coverage = {"layer": layer_idx, "gate_up": [], "down": []}
    for target, weight_name in (
        ("gate_up", "gate_up_proj"),
        ("down", "down_proj"),
    ):
        chunks, hits = _collect_expert_inputs(
            layer=layer,
            experts=experts,
            inps=inps,
            outs=outs,
            nsamples=nsamples,
            rope=rope,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_device=layer_device,
            target=target,
        )
        weight_bank = getattr(experts, weight_name)
        for expert_idx in range(experts.num_experts):
            n_hits = hits[expert_idx]
            used_gptq = n_hits >= min_expert_calibration_hits
            if not used_gptq:
                logging.warning(
                    "layer %d expert %d %s has %d calibration hits; using RTN",
                    layer_idx,
                    expert_idx,
                    weight_name,
                    n_hits,
                )
            quantized = _quantize_expert_slice(
                weight=weight_bank[expert_idx],
                chunks=chunks[expert_idx] if used_gptq else [],
                fmt=fmt,
                weight_config=weight_config,
                quantile_search=quantile_search,
                clip_search_y=clip_search_y,
                cali_batch_size=cali_batch_size,
                layer_name=(
                    f"layers{layer_idx}.mlp.experts.{expert_idx}.{weight_name}"
                ),
            )
            weight_bank.data[expert_idx].copy_(quantized)
            coverage[target].append(
                {"expert": expert_idx, "hits": n_hits, "gptq": used_gptq}
            )
    return coverage


@torch.no_grad()
def run_gptq(network, gptq_config):
    """
    Run GPTQ weight optimization on all nn.Linear layers in decoder blocks.

    Quantized weights are written back in-place to the existing nn.Linear
    modules so that the subsequent module-replacement pass can pick them up.

    Args:
        network: HuggingFace causal-LM model (e.g. LlamaForCausalLM).
        gptq_config: Dict with keys:
            model_name: str - HF model name (for tokenizer).
            device: str - e.g. "cuda:0".
            dataset: str - "wikitext2" | "c4" | "ptb".
            nsamples: int - calibration samples (default 128).
            seqlen: int - sequence length (default 2048).
            format: str - "mxfp" | "mxint".
            weight_config: dict - Mase-style weight config, e.g.
                {"weight_block_size": 32, "weight_exponent_width": 2, "weight_frac_width": 1}
            phase: str - "both" (default, legacy in-place), "decode"
                (GPTQ result becomes the decode weight bank, FP restored for
                prefill), or "prefill" (in-place + FP snapshot for fp-only
                decode).
            quantile_search: bool (default True).
            clip_search_y: bool (default False).
            cali_batch_size: int (default 32).
            checkpoint_dir: str | None.
            hf_token: str | None.

    Returns:
        network with GPTQ-optimized weights (still nn.Linear modules).
    """
    logging.info("-----GPTQ Quantization-----")

    model_name = gptq_config["model_name"]
    dev = gptq_config.get("device", "cuda:0")
    dataset = gptq_config.get("dataset", "wikitext2")
    nsamples = gptq_config.get("nsamples", 128)
    seqlen = gptq_config.get("seqlen", 2048)
    fmt = gptq_config["format"]
    weight_config = gptq_config["weight_config"]
    phase = gptq_config.get("phase", "both")
    if phase not in _VALID_GPTQ_PHASES:
        raise ValueError(
            f"Unsupported gptq phase {phase!r}; expected one of {_VALID_GPTQ_PHASES}."
        )
    quantile_search = gptq_config.get("quantile_search", True)
    clip_search_y = gptq_config.get("clip_search_y", False)
    cali_batch_size = gptq_config.get("cali_batch_size", 32)
    checkpoint_dir = gptq_config.get("checkpoint_dir", None)
    hf_token = gptq_config.get("hf_token", None)
    max_layers = gptq_config.get("max_layers", None)
    device_map_aware = bool(gptq_config.get("device_map_aware", False))
    min_expert_calibration_hits = int(
        gptq_config.get("min_expert_calibration_hits", 1)
    )
    if min_expert_calibration_hits < 1:
        raise ValueError("min_expert_calibration_hits must be positive")

    if phase != "both":
        # FP snapshots must precede checkpoint resume: resume overwrites
        # network weights with already-quantised checkpoints.
        _snapshot_fp_linear_weights(network.model.layers)

    # Handle checkpoint resuming
    start_layer = 0
    if checkpoint_dir is not None:
        max_quantized_layer = auto_load_quantized_layers(network, checkpoint_dir)
        if max_quantized_layer >= 0:
            start_layer = max_quantized_layer + 1
            logging.info(f"Resuming GPTQ from layer {start_layer}")

        if start_layer == len(network.model.layers):
            logging.info("All layers already quantized, skipping GPTQ")
            _finalize_gptq_phase(network, phase)
            return network

    # Load calibration data
    dataloader = get_loaders(
        dataset,
        nsamples=nsamples,
        seed=0,
        seqlen=seqlen,
        model=model_name,
        hf_token=hf_token,
    )

    # Disable kv cache
    use_cache = network.config.use_cache
    network.config.use_cache = False

    layers = network.model.layers

    if device_map_aware:
        embed_device = _module_device(
            network.model.embed_tokens, "model.embed_tokens"
        )
        _module_device(layers[0], "model.layers.0")
        rope = network.model.rotary_emb
    else:
        embed_device = torch.device(dev)
        network.model.embed_tokens = network.model.embed_tokens.to(dev)
        network.model.norm = network.model.norm.to(dev)
        rope = network.model.rotary_emb.to(dev)
        layers[0] = layers[0].to(dev)

    dtype = next(iter(network.parameters())).dtype

    buffer_device = torch.device("cpu") if device_map_aware else torch.device(dev)
    inps = torch.zeros(
        (nsamples, seqlen, network.config.hidden_size),
        dtype=dtype,
        device=buffer_device,
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if cache["i"] >= nsamples:
                raise _CalibrationCaptureComplete
            inps[cache["i"]].copy_(
                inp.squeeze(0).detach().to(device=inps.device, dtype=inps.dtype)
            )
            cache["i"] += 1
            cache["attention_mask"] = _move_to_device(
                kwargs.get("attention_mask"), inps.device
            )
            cache["position_ids"] = _move_to_device(
                kwargs.get("position_ids"), inps.device
            )
            raise _CalibrationCaptureComplete

    original_first_layer = layers[0]
    layers[0] = Catcher(original_first_layer)
    try:
        for batch in dataloader:
            if cache["i"] >= nsamples:
                break
            try:
                network(batch[0].to(embed_device))
            except _CalibrationCaptureComplete:
                pass
    finally:
        layers[0] = original_first_layer
    torch.cuda.empty_cache()

    collected = int(cache["i"])
    if collected <= 0:
        raise RuntimeError("No calibration samples collected from dataloader.")
    if collected < nsamples:
        logging.warning(
            "GPTQ requested nsamples=%d but only collected=%d; "
            "continuing with collected samples.",
            nsamples, collected,
        )
        nsamples = collected
        inps = inps[:nsamples]
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    end_layer = (
        len(layers)
        if max_layers is None
        else min(start_layer + max_layers, len(layers))
    )
    logging.info(
        f"GPTQ: quantizing layers {start_layer} to {end_layer - 1} (of {len(layers)} total)"
    )

    expert_coverage = []
    for i in range(start_layer, end_layer):
        print(f"\nLayer {i}:", flush=True, end=" ")
        if device_map_aware:
            layer = layers[i]
            layer_device = _module_device(layer, f"model.layers.{i}")
        else:
            layer = layers[i].to(dev)
            layer_device = torch.device(dev)
        full = find_qlayers(layer, layers=[torch.nn.Linear])

        for names in _DENSE_SEQUENTIAL:
            subset = {name: full[name] for name in names if name in full}
            if not subset:
                continue

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                gptq[name] = GPTQ(subset[name])

            pre_act = []

            def make_pre_hook():
                def pre_hook(_, inp):
                    if clip_search_y:
                        pre_act.append(inp[0].detach().cpu())

                return pre_hook

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            first_module = next(iter(subset.values()))
            handles.append(first_module.register_forward_pre_hook(make_pre_hook()))

            for j in range(nsamples):
                _run_layer_sample(
                    layer=layer,
                    sample=inps[j],
                    outs=outs,
                    sample_idx=j,
                    rope=rope,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    layer_device=layer_device,
                )

            activation = (
                torch.cat(pre_act, dim=0).to(layer_device) if pre_act else None
            )

            for h in handles:
                h.remove()

            for name in subset:
                quantized_w = gptq[name].fasterquant(
                    activation=activation,
                    fmt=fmt,
                    weight_config=weight_config,
                    percdamp=0.01,
                    cali_batch_size=cali_batch_size,
                    layer_name=f"layers{i}.{name}",
                    quant_search=quantile_search,
                )

                # Write quantized weights back in-place
                assert quantized_w.shape == gptq[name].layer.weight.shape
                gptq[name].layer.weight.data.copy_(quantized_w)
                gptq[name].free()

        if getattr(network.config, "model_type", None) == "qwen3_moe":
            coverage = _quantize_qwen3_moe_experts(
                layer=layer,
                layer_idx=i,
                inps=inps,
                outs=outs,
                nsamples=nsamples,
                rope=rope,
                attention_mask=attention_mask,
                position_ids=position_ids,
                fmt=fmt,
                weight_config=weight_config,
                quantile_search=quantile_search,
                clip_search_y=clip_search_y,
                cali_batch_size=cali_batch_size,
                layer_device=layer_device,
                min_expert_calibration_hits=min_expert_calibration_hits,
            )
            if coverage is not None:
                expert_coverage.append(coverage)

        # Forward pass with quantized weights to get inputs for next layer
        for j in range(nsamples):
            _run_layer_sample(
                layer=layer,
                sample=inps[j],
                outs=outs,
                sample_idx=j,
                rope=rope,
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_device=layer_device,
            )

        if not device_map_aware:
            layers[i] = layer.cpu()
            del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        if checkpoint_dir is not None:
            save_layer_checkpoint(network, i, checkpoint_dir)

    network.config.use_cache = use_cache
    network._mase_gptq_expert_coverage = expert_coverage
    _finalize_gptq_phase(network, phase)
    cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")

    return network
