import torch
from torch import Tensor
import torch.nn.functional as F


def opt_patched_fn_get_max_float_tensor(attn_weights: Tensor) -> Tensor:
    return torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)


def opt_patched_fn_get_tensor_dtype(tensor: Tensor) -> Tensor:
    return tensor.dtype


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int = None):
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


def opt_patched_fn_prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

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


def opt_patched_shape_assertion_0(
    attn_weights: Tensor, bsz: int, num_heads: int, tgt_len: int, src_len: int
):
    assert attn_weights.size() == (
        bsz * num_heads,
        tgt_len,
        src_len,
    ), f"Attention weights should be of size {(bsz * num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"


def opt_patched_shape_assertion_1(
    attention_mask: Tensor, bsz: int, tgt_len: int, src_len: int
):
    assert attention_mask.size() == (
        bsz,
        1,
        tgt_len,
        src_len,
    ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"


def opt_patched_shape_assertion_2(layer_head_mask: Tensor, num_heads: int):
    assert layer_head_mask.size() == (
        num_heads,
    ), f"Head mask for a single layer should be of size {(num_heads,)}, but is {layer_head_mask.size()}"


def opt_patched_shape_assertion_3(
    attn_output: Tensor, bsz: int, num_heads: int, tgt_len: int, head_dim: int
):
    assert attn_output.size() == (
        bsz * num_heads,
        tgt_len,
        head_dim,
    ), f"`attn_output` should be of size {(bsz, num_heads, tgt_len, head_dim)}, but is {attn_output.size()}"


def opt_patched_shape_assertion_4(attention_mask: Tensor, mask_seq_length: int):
    assert (
        attention_mask.shape[1] == mask_seq_length
    ), f"Attention mask shape mismatch: {attention_mask.shape[1]} != {mask_seq_length}"


def opt_patched_shape_assertion_5(head_mask: Tensor, num_layers):
    for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
        if attn_mask is not None:
            if attn_mask.size()[0] != (num_layers):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {(num_layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )


def opt_patched_fn_calculate_causal_lm_loss(logits, labels, vocab_size: int):
    loss = None
    if labels is not None:
        # move labels to correct device to enable model parallelism
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        # PATCH FIX: use F.cross_entropy instead of nn.CrossEntropyLoss
        # loss = F.cross_entropy(
        #     shift_logits.view(-1, vocab_size),
        #     shift_labels.view(-1),
        # )
    return 0
    # return loss
