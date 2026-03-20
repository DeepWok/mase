import torch
import tqdm
from typing import Iterable

from chop.nn.quantized.rotation import (
    apply_exact_had_to_linear,
    random_hadamard_matrix,
)


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    """Mean-center and rotate embedding weights using orthogonal matrix Q."""
    weight = model.model.embed_tokens.weight

    weight_mean = weight.data.mean(dim=-1, keepdim=True)
    weight.data -= weight_mean

    assert (
        Q.shape[0] == Q.shape[1] == weight.shape[1]
    ), f"Q must be square and match embedding dim ({weight.shape[1]}), but got {Q.shape}"

    Q = Q.to(weight.device)
    rotated = (weight.data.double() @ Q).to(dtype=weight.dtype)
    weight.data.copy_(rotated)


def rotate_lm_head(model, Q: torch.Tensor) -> None:
    """Rotate the LM head weights using orthogonal matrix Q."""
    weight = model.lm_head.weight
    Q = Q.to(weight.device)
    rotated = (weight.data.double() @ Q).to(dtype=weight.dtype)
    weight.data.copy_(rotated)


def rotate_attention_inputs(
    attn_layer, head_dim, Q: torch.Tensor, online_rotate: bool = True
) -> None:
    """Rotate q/k/v projection weights. Optionally apply Hadamard on v_proj output."""
    for proj in [attn_layer.q_proj, attn_layer.k_proj, attn_layer.v_proj]:
        weight = proj.weight
        Q = Q.to(weight.device)
        rotated = (weight.data.double() @ Q).to(dtype=weight.dtype)
        weight.data.copy_(rotated)

    if online_rotate:
        apply_exact_had_to_linear(attn_layer.v_proj, had_dim=head_dim, output=True)


def rotate_attention_output(
    attn_layer, Q: torch.Tensor, online_rotate: bool = True
) -> None:
    """Rotate o_proj weights using Q.T. Optionally apply Hadamard on input side."""
    weight = attn_layer.o_proj.weight
    Q = Q.to(weight.device)
    rotated = (Q.T @ weight.data.double()).to(dtype=weight.dtype)
    weight.data.copy_(rotated)

    if online_rotate:
        apply_exact_had_to_linear(attn_layer.o_proj, had_dim=-1, output=False)


def rotate_mlp_input(mlp, Q: torch.Tensor) -> None:
    """Rotate up_proj and gate_proj weights."""
    for ln in [mlp.up_proj, mlp.gate_proj]:
        weight = ln.weight
        Q = Q.to(weight.device)
        rotated = (weight.data.double() @ Q).to(dtype=weight.dtype)
        weight.data.copy_(rotated)


def rotate_mlp_output(mlp, Q: torch.Tensor, online_rotate: bool = True) -> None:
    """Rotate down_proj weights using Q.T. Optionally apply Hadamard on input side."""
    down_proj = mlp.down_proj
    weight = down_proj.weight
    Q = Q.to(weight.device)
    rotated = (Q.T @ weight.data.double()).to(dtype=weight.dtype)
    weight.data.copy_(rotated)

    if online_rotate:
        apply_exact_had_to_linear(down_proj, had_dim=-1, output=False)


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: Iterable[torch.nn.Linear]
) -> None:
    """Fuse RMSNorm scale (and optional bias) into linear layer weights."""
    for linear in linear_layers:
        weight = linear.weight.data
        dtype = weight.dtype

        gamma = layernorm.weight.data.double().to(weight.device)
        W_fused = (weight.double() * gamma).to(dtype)
        linear.weight.data.copy_(W_fused)

        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            beta = layernorm.bias.data.double()
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            b_fused = (linear.bias.data.double() + weight.double() @ beta).to(dtype)
            linear.bias.data.copy_(b_fused)


def fuse_rms_norms(model):
    """Fuse all RMSNorm scales into downstream linear layers (Llama/Qwen3)."""
    decoder_layers = model.model.layers
    for layer in decoder_layers:
        fuse_ln_linear(
            layer.input_layernorm,
            [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
        )
        fuse_ln_linear(
            layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
        )

    fuse_ln_linear(model.model.norm, [model.lm_head])


def replace_rms_norms(model: torch.nn.Module) -> None:
    """Replace all LlamaRMSNorm/Qwen3RMSNorm with unfused RMSN (no learnable weights)."""
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    rms_norm_classes = (LlamaRMSNorm, Qwen3RMSNorm)

    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, rms_norm_classes):
                hidden_dim = child.weight.shape[0]
                device = child.weight.device
                new_norm = RMSN(mean_dim=hidden_dim, eps=child.variance_epsilon).to(
                    device
                )
                setattr(module, child_name, new_norm)


class RMSN(torch.nn.Module):
    """RMSNorm without learnable weights (assumes weights already fused into linears)."""

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def _get_hadamard(size: int) -> torch.Tensor:
    """Generate random Hadamard matrix for offline rotation."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return random_hadamard_matrix(size, device)


@torch.inference_mode()
def rotate_llama(model, online_rotate=False):
    """Apply QuaRot rotation to a Llama model.

    Rotates all weights using a random Hadamard matrix.
    If online_rotate=True, also bakes in exact Hadamard transforms for online
    rotation during inference.

    NOTE: Call fuse_rms_norms() and replace_rms_norms() before this if needed.
    """
    # Untie lm_head from embed_tokens if they share the same tensor,
    # otherwise fusing model.norm into lm_head corrupts the embeddings.
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())

    num_heads = model.config.num_attention_heads
    hidden_dim = model.config.hidden_size
    head_dim = hidden_dim // num_heads

    Q = _get_hadamard(hidden_dim)

    rotate_embeddings(model, Q)
    decoder_layers = model.model.layers
    for _, layer in enumerate(
        tqdm.tqdm(decoder_layers, unit="LlamaDecoderLayer", desc="Rotating")
    ):
        rotate_attention_inputs(layer.self_attn, head_dim, Q, online_rotate)
        rotate_attention_output(layer.self_attn, Q, online_rotate)
        rotate_mlp_input(layer.mlp, Q)
        rotate_mlp_output(layer.mlp, Q, online_rotate)
    rotate_lm_head(model, Q)


@torch.inference_mode()
def rotate_qwen3(model, online_rotate=False):
    """Apply QuaRot rotation to a Qwen3 model.

    Same structure as rotate_llama — Qwen3 shares the same decoder layout
    (self_attn with q/k/v/o_proj, mlp with gate/up/down_proj, embed_tokens).

    NOTE: Call fuse_rms_norms() and replace_rms_norms() before this if needed.
    """
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())

    num_heads = model.config.num_attention_heads
    hidden_dim = model.config.hidden_size
    head_dim = getattr(model.config, "head_dim", hidden_dim // num_heads)

    Q = _get_hadamard(hidden_dim)

    rotate_embeddings(model, Q)
    decoder_layers = model.model.layers
    for _, layer in enumerate(
        tqdm.tqdm(decoder_layers, unit="Qwen3DecoderLayer", desc="Rotating")
    ):
        rotate_attention_inputs(layer.self_attn, head_dim, Q, online_rotate)
        rotate_attention_output(layer.self_attn, Q, online_rotate)
        rotate_mlp_input(layer.mlp, Q)
        rotate_mlp_output(layer.mlp, Q, online_rotate)
    rotate_lm_head(model, Q)
