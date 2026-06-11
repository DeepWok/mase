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


_DENSE_SEQUENTIAL = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.o_proj"],
    ["mlp.up_proj", "mlp.gate_proj"],
    ["mlp.down_proj"],
]


class _WeightView:
    """Minimal GPTQ adapter for tensors that are not nn.Linear modules."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight


def _activation_tensor(chunks: list[torch.Tensor]) -> torch.Tensor | None:
    if not chunks:
        return None
    return torch.cat(chunks, dim=0).unsqueeze(0)


def _quantize_uncalibrated_weight(
    weight: torch.Tensor,
    *,
    fmt: str,
    weight_config: dict,
    quantile_search: bool,
) -> torch.Tensor:
    return quantize_tensor(
        weight.data,
        block_dim=1,
        fmt=fmt,
        config=weight_config,
        quantile_search=quantile_search,
    )


def _quantize_gptq_weight(
    weight: torch.Tensor,
    gptq: GPTQ,
    *,
    activation: torch.Tensor | None,
    fmt: str,
    weight_config: dict,
    quantile_search: bool,
    clip_search_y: bool,
    cali_batch_size: int,
    layer_name: str,
) -> torch.Tensor:
    quantized_w = gptq.fasterquant(
        activation=activation if clip_search_y else None,
        fmt=fmt,
        weight_config=weight_config,
        percdamp=0.01,
        cali_batch_size=cali_batch_size,
        layer_name=layer_name,
        quant_search=quantile_search,
    )
    assert quantized_w.shape == weight.shape
    gptq.free()
    return quantized_w


def _run_layer_forward(layer, x, attention_mask, position_embeddings):
    return layer(
        x,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )[0]


def _quantize_dense_linears(
    *,
    layer,
    layer_idx: int,
    inps: torch.Tensor,
    outs: torch.Tensor,
    nsamples: int,
    rope,
    attention_mask,
    position_ids,
    fmt: str,
    weight_config: dict,
    quantile_search: bool,
    clip_search_y: bool,
    cali_batch_size: int,
) -> None:
    full = find_qlayers(layer, layers=[torch.nn.Linear])

    for names in _DENSE_SEQUENTIAL:
        subset = {name: full[name] for name in names if name in full}
        if not subset:
            continue

        gptq = {}
        for name, module in subset.items():
            print(f"{name}", end="  ", flush=True)
            gptq[name] = GPTQ(module)

        pre_act = []

        def make_pre_hook():
            def pre_hook(_, inp):
                pre_act.append(inp[0])

            return pre_hook

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name, module in subset.items():
            handles.append(module.register_forward_hook(add_batch(name)))
        first_module = next(iter(subset.values()))
        handles.append(first_module.register_forward_pre_hook(make_pre_hook()))

        for j in range(nsamples):
            x = inps[j].unsqueeze(0)
            cos, sin = rope(x, position_ids)
            outs[j] = _run_layer_forward(layer, x, attention_mask, (cos, sin))

        activation = torch.cat(pre_act, dim=0) if pre_act else None

        for h in handles:
            h.remove()

        for name, module in subset.items():
            quantized_w = _quantize_gptq_weight(
                module.weight,
                gptq[name],
                activation=activation,
                fmt=fmt,
                weight_config=weight_config,
                quantile_search=quantile_search,
                clip_search_y=clip_search_y,
                cali_batch_size=cali_batch_size,
                layer_name=f"layers{layer_idx}.{name}",
            )
            module.weight.data.copy_(quantized_w)


def _get_qwen3_moe_experts(layer):
    mlp = getattr(layer, "mlp", None)
    experts = getattr(mlp, "experts", None)
    if experts is None:
        return None
    if not hasattr(experts, "gate_up_proj") or not hasattr(experts, "down_proj"):
        return None
    return experts


def _collect_qwen3_moe_expert_batches(
    *,
    layer,
    experts,
    inps: torch.Tensor,
    outs: torch.Tensor,
    nsamples: int,
    rope,
    attention_mask,
    position_ids,
    collect_activations: bool,
    target: str,
):
    if target not in {"gate_up", "down"}:
        raise ValueError(f"Unsupported Qwen3-MoE expert GPTQ target: {target!r}")

    gptq = [None for _ in range(experts.num_experts)]
    acts = [[] for _ in range(experts.num_experts)]
    hits = [0 for _ in range(experts.num_experts)]

    def hook(_, inp, _out):
        hidden_states, top_k_index, _top_k_weights = inp
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=experts.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_hit_idx in expert_hit:
                expert_idx = int(expert_hit_idx[0].item())
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                del top_k_pos
                current_state = hidden_states[token_idx]
                if current_state.numel() == 0:
                    continue

                if target == "gate_up":
                    batch = current_state
                    weight = experts.gate_up_proj[expert_idx]
                else:
                    gate, up = F.linear(current_state, experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                    batch = experts.act_fn(gate) * up
                    weight = experts.down_proj[expert_idx]

                if gptq[expert_idx] is None:
                    gptq[expert_idx] = GPTQ(_WeightView(weight))
                gptq[expert_idx].add_batch(batch.unsqueeze(0).data, None)
                hits[expert_idx] += int(batch.shape[0])

                if collect_activations:
                    acts[expert_idx].append(batch.detach())

    handle = experts.register_forward_hook(hook)
    try:
        for j in range(nsamples):
            x = inps[j].unsqueeze(0)
            cos, sin = rope(x, position_ids)
            outs[j] = _run_layer_forward(layer, x, attention_mask, (cos, sin))
    finally:
        handle.remove()

    return {
        "gptq": gptq,
        "acts": acts,
        "hits": hits,
    }


def _quantize_qwen3_moe_experts(
    *,
    layer,
    layer_idx: int,
    inps: torch.Tensor,
    outs: torch.Tensor,
    nsamples: int,
    rope,
    attention_mask,
    position_ids,
    fmt: str,
    weight_config: dict,
    quantile_search: bool,
    clip_search_y: bool,
    cali_batch_size: int,
) -> bool:
    experts = _get_qwen3_moe_experts(layer)
    if experts is None:
        return False

    print("mlp.experts.gate_up_proj", end="  ", flush=True)
    gate_up_batches = _collect_qwen3_moe_expert_batches(
        layer=layer,
        experts=experts,
        inps=inps,
        outs=outs,
        nsamples=nsamples,
        rope=rope,
        attention_mask=attention_mask,
        position_ids=position_ids,
        collect_activations=clip_search_y,
        target="gate_up",
    )

    for expert_idx in range(experts.num_experts):
        gate_up_weight = experts.gate_up_proj[expert_idx]
        if gate_up_batches["hits"][expert_idx] > 0:
            gate_up_quantized = _quantize_gptq_weight(
                gate_up_weight,
                gate_up_batches["gptq"][expert_idx],
                activation=_activation_tensor(gate_up_batches["acts"][expert_idx]),
                fmt=fmt,
                weight_config=weight_config,
                quantile_search=quantile_search,
                clip_search_y=clip_search_y,
                cali_batch_size=cali_batch_size,
                layer_name=f"layers{layer_idx}.mlp.experts.{expert_idx}.gate_up_proj",
            )
        else:
            logging.warning(
                "Layer %d expert %d gate_up_proj had no calibration hits; using direct tensor quantization.",
                layer_idx,
                expert_idx,
            )
            gate_up_quantized = _quantize_uncalibrated_weight(
                gate_up_weight,
                fmt=fmt,
                weight_config=weight_config,
                quantile_search=quantile_search,
            )
        experts.gate_up_proj.data[expert_idx].copy_(gate_up_quantized)

    print("mlp.experts.down_proj", end="  ", flush=True)
    down_batches = _collect_qwen3_moe_expert_batches(
        layer=layer,
        experts=experts,
        inps=inps,
        outs=outs,
        nsamples=nsamples,
        rope=rope,
        attention_mask=attention_mask,
        position_ids=position_ids,
        collect_activations=clip_search_y,
        target="down",
    )

    for expert_idx in range(experts.num_experts):
        down_weight = experts.down_proj[expert_idx]
        if down_batches["hits"][expert_idx] > 0:
            down_quantized = _quantize_gptq_weight(
                down_weight,
                down_batches["gptq"][expert_idx],
                activation=_activation_tensor(down_batches["acts"][expert_idx]),
                fmt=fmt,
                weight_config=weight_config,
                quantile_search=quantile_search,
                clip_search_y=clip_search_y,
                cali_batch_size=cali_batch_size,
                layer_name=f"layers{layer_idx}.mlp.experts.{expert_idx}.down_proj",
            )
        else:
            logging.warning(
                "Layer %d expert %d down_proj had no calibration hits; using direct tensor quantization.",
                layer_idx,
                expert_idx,
            )
            down_quantized = _quantize_uncalibrated_weight(
                down_weight,
                fmt=fmt,
                weight_config=weight_config,
                quantile_search=quantile_search,
            )
        experts.down_proj.data[expert_idx].copy_(down_quantized)

    return True


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
    quantile_search = gptq_config.get("quantile_search", True)
    clip_search_y = gptq_config.get("clip_search_y", False)
    cali_batch_size = gptq_config.get("cali_batch_size", 32)
    checkpoint_dir = gptq_config.get("checkpoint_dir", None)
    hf_token = gptq_config.get("hf_token", None)
    max_layers = gptq_config.get("max_layers", None)

    # Handle checkpoint resuming
    start_layer = 0
    if checkpoint_dir is not None:
        max_quantized_layer = auto_load_quantized_layers(network, checkpoint_dir)
        if max_quantized_layer >= 0:
            start_layer = max_quantized_layer + 1
            logging.info(f"Resuming GPTQ from layer {start_layer}")

        if start_layer == len(network.model.layers):
            logging.info("All layers already quantized, skipping GPTQ")
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

    # Move embedding + norm + rope to device
    network.model.embed_tokens = network.model.embed_tokens.to(dev)
    network.model.norm = network.model.norm.to(dev)
    rope = network.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(network.parameters())).dtype

    inps = torch.zeros(
        (nsamples, seqlen, network.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if cache["i"] >= nsamples:
                raise ValueError
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        if cache["i"] >= nsamples:
            break
        try:
            network(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
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

    for i in range(start_layer, end_layer):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)
        _quantize_dense_linears(
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
        )

        if getattr(network.config, "model_type", None) == "qwen3_moe":
            _quantize_qwen3_moe_experts(
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
            )

        # Forward pass with quantized weights to get inputs for next layer
        for j in range(nsamples):
            x = inps[j].unsqueeze(0)
            cos, sin = network.model.rotary_emb(x, position_ids)
            outs[j] = _run_layer_forward(layer, x, attention_mask, (cos, sin))

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        if checkpoint_dir is not None:
            save_layer_checkpoint(network, i, checkpoint_dir)

    network.config.use_cache = use_cache
    cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")

    return network
