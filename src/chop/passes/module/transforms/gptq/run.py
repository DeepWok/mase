"""
Main GPTQ orchestration: run_gptq(network, gptq_config).

Ported from Coprocessor_for_Llama/acc_simulator/gptq/quant.py,
adapted to use Mase config dicts and write quantized weights
back in-place to nn.Linear modules (no module replacement here).
"""

import logging

import torch
import torch.nn as nn

from .gptq import GPTQ
from .utils import find_qlayers, cleanup_memory
from .data_utils import get_loaders
from .checkpoint import save_layer_checkpoint, auto_load_quantized_layers
from chop.nn.quantized.modules.phase_config import (
    DECODE_FP_BIAS_ATTR,
    DECODE_FP_WEIGHT_ATTR,
    GPTQ_DECODE_WEIGHT_ATTR,
)

_VALID_GPTQ_PHASES = ("both", "decode", "prefill")


def _snapshot_fp_linear_weights(layers) -> None:
    """Snapshot FP weights of every decoder linear before GPTQ mutates them.

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
            if not isinstance(module, nn.Linear):
                continue
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

    sequential = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

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
        full = find_qlayers(layer, layers=[torch.nn.Linear])

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                gptq[name] = GPTQ(subset[name])

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
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            handles.append(subset[name].register_forward_pre_hook(make_pre_hook()))

            for j in range(nsamples):
                x = inps[j].unsqueeze(0)
                cos, sin = rope(x, position_ids)
                outs[j] = layer(
                    x,
                    attention_mask=attention_mask,
                    position_embeddings=(cos, sin),
                )[0]

            pre_act = torch.cat(pre_act, dim=0)

            for h in handles:
                h.remove()

            for name in subset:
                quantized_w = gptq[name].fasterquant(
                    activation=pre_act if clip_search_y else None,
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

        # Forward pass with quantized weights to get inputs for next layer
        for j in range(nsamples):
            x = inps[j].unsqueeze(0)
            cos, sin = network.model.rotary_emb(x, position_ids)
            outs[j] = layer(
                x,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
            )[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        if checkpoint_dir is not None:
            save_layer_checkpoint(network, i, checkpoint_dir)

    network.config.use_cache = use_cache
    _finalize_gptq_phase(network, phase)
    cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")

    return network
