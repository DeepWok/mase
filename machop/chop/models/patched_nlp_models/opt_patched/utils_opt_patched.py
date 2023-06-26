from typing import Optional, Tuple

import torch
from torch import Tensor



def OPTAttention_self_shape(
    tensor: Tensor, seq_len: int, bsz: int, num_heads: int, head_dim: int
) -> Tensor:
    """
    reshape and permute the Tensor for matmul
    [B, N, h*d_head] -> [B, N, h, d_head] -> [B, h, N, d_head]

    replaces `OPTAttention._shape` method
    """
    return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


def OPTAttention_reshape_qkv_back_for_bmm(
    query_states: Tensor,
    key_states: Tensor,
    value_states: Tensor,
    proj_shape: int,
    tgt_len: int,
    bsz: int,
    num_heads: int,
    head_dim: int,
) -> Tuple[Tensor]:
    query_states = OPTAttention_self_shape(
        query_states, tgt_len, bsz, num_heads, head_dim
    ).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)
    return query_states, key_states, value_states


def OPTAttention_attn_weights_shape_check(
    attn_weights: Tensor, bsz: int, num_heads: int, tgt_len: int, src_len: int
) -> bool:
    if attn_weights.size() != (bsz * num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )


def OPTAttention_attention_mask_shape_check(
    attention_mask: Tensor, bsz: int, tgt_len: int, src_len: int
) -> bool:
    if attention_mask.size() != (bsz, 1, tgt_len, src_len):
        raise ValueError(
            f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
        )


def OPTAttention_attention_get_dtype_min(attn_weights: Tensor) -> Tensor:
    return torch.tensor(torch.finfo(attn_weights.dtype).min)


def OPTAttention_attn_weight_dtype_check(attn_weights: Tensor) -> bool:
    assert (
        attn_weights.dtype != torch.float16
    ), "FP16 is not supported for patched OPTAttention"


def OPTAttention_layer_head_mask_shape_check(
    layer_head_mask: Tensor, num_heads: int
) -> bool:
    if layer_head_mask.size() != (num_heads,):
        raise ValueError(
            f"Head mask for a single layer should be of size {(num_heads,)}, but is"
            f" {layer_head_mask.size()}"
        )


def OPTAttention_attn_output_shape_check(
    attn_output: Tensor, bsz: int, num_heads: int, tgt_len: int, head_dim: int
) -> bool:
    if attn_output.size() != (bsz * num_heads, tgt_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, tgt_len, head_dim)}, but is"
            f" {attn_output.size()}"
        )


# @mark_as_leaf_func
# def OPTAttention_construct_proj_shape(
#     bsz: int, num_heads: int, head_dim: int
# ) -> Tuple[int]:
#     proj_shape = (bsz * num_heads, -1, head_dim)
#     return proj_shape


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


# @mark_as_leaf_func
# def OPTDecoder_view_input_ids(input_ids, input_shape):
#     input_ids = input_ids.view(-1, input_shape[-1])
#     return input_ids


def OPTDecoder_self_prepare_decoder_attention(
    attention_mask: Tensor,
    input_shape,
    inputs_embeds: Tensor,
    past_key_values_length: int,
) -> Tensor:
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            past_key_values_length=past_key_values_length,
        ).to(inputs_embeds.device)

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def OPTDecoder_check_head_mask(head_mask, decoder_layers) -> bool:
    for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
        if attn_mask is not None:
            if attn_mask.size()[0] != (len(decoder_layers)):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(decoder_layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )


def OPTForCasualLM_compute_loss(logits, labels, self_config_vocab_size):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, self_config_vocab_size), shift_labels.view(-1)
    )
    # loss = self_loss_fct(
    #     shift_logits.view(-1, self_config_vocab_size), shift_labels.view(-1)
    # )
    return loss


opt_leaf_funcs = [
    ("OPTAttention_attn_weights_shape_check", OPTAttention_attn_weights_shape_check),
    ("OPTAttention_attention_mask_shape_check", OPTAttention_attention_mask_shape_check),
    ("OPTAttention_attention_get_dtype_min", OPTAttention_attention_get_dtype_min),
    ("OPTAttention_attn_weight_dtype_check", OPTAttention_attn_weight_dtype_check),
    ("OPTAttention_layer_head_mask_shape_check", OPTAttention_layer_head_mask_shape_check),
    ("OPTAttention_attn_output_shape_check", OPTAttention_attn_output_shape_check),
    ("OPTDecoder_self_prepare_decoder_attention", OPTDecoder_self_prepare_decoder_attention),
    ("OPTDecoder_check_head_mask", OPTDecoder_check_head_mask),
    ("OPTForCasualLM_compute_loss", OPTForCasualLM_compute_loss)
]
