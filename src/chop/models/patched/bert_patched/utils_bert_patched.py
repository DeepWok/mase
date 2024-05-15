import torch
from torch import Tensor
from transformers.modeling_utils import ModuleUtilsMixin


def BertSelfAttention_transpose_for_scores(
    x, self_num_attention_heads: int, self_attention_head_size: int
) -> Tensor:
    new_x_shape = (
        x.size(0),
        x.size(1),
        self_num_attention_heads,
        self_attention_head_size,
    )
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)


def BertSelfAttention_add_relative_position_embedding(
    self_position_embedding_type: str,
    query_layer: Tensor,
    key_layer: Tensor,
    hidden_states: Tensor,
    self_distance_embedding: torch.nn.Embedding,
    self_max_position_embeddings: int,
) -> Tensor:
    query_length, key_length = query_layer.shape[2], key_layer.shape[2]
    position_ids_l = torch.arange(
        query_length, dtype=torch.long, device=hidden_states.device
    ).view(-1, 1)
    position_ids_r = torch.arange(
        key_length, dtype=torch.long, device=hidden_states.device
    ).view(1, -1)
    distance = position_ids_l - position_ids_r

    positional_embedding = self_distance_embedding(
        distance + self_max_position_embeddings - 1
    )
    positional_embedding = positional_embedding.to(
        dtype=query_layer.dtype
    )  # fp16 compatibility

    if self_position_embedding_type == "relative_key":
        relative_position_scores = torch.einsum(
            "bhld,lrd->bhlr", query_layer, positional_embedding
        )
        attention_scores = attention_scores + relative_position_scores
    elif self_position_embedding_type == "relative_key_query":
        relative_position_scores_query = torch.einsum(
            "bhld,lrd->bhlr", query_layer, positional_embedding
        )
        relative_position_scores_key = torch.einsum(
            "bhrd,lrd->bhlr", key_layer, positional_embedding
        )
        attention_scores = (
            attention_scores
            + relative_position_scores_query
            + relative_position_scores_key
        )
    return attention_scores


def BertSelfAttention_get_new_context_layer_shape(
    context_layer: Tensor,
    self_all_head_size: int,
):
    new_context_layer_shape = (
        context_layer.size(0),
        context_layer.size(1),
        self_all_head_size,
    )
    return new_context_layer_shape


def BertModel_get_input_shape_batch_size_seq_length_and_device(
    input_ids: Tensor,
):
    input_shape = input_ids.size()
    batch_size, seq_length = input_shape
    device = input_ids.device
    return input_shape, batch_size, seq_length, device


def BertModel_create_default_attention_mask(batch_size: int, seq_length: int, device):
    attention_mask = torch.ones((batch_size, seq_length), device=device)
    return attention_mask


def BertModel_create_default_token_type_ids(
    self_embeddings,
    seq_length: int,
    batch_size: int,
    input_shape,
    device,
):
    if hasattr(self_embeddings, "token_type_ids"):
        buffered_token_type_ids = self_embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
            batch_size, seq_length
        )
        token_type_ids = buffered_token_type_ids_expanded
    else:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    return token_type_ids


def BertModel_get_extended_attention_mask(
    attention_mask: Tensor,
    input_shape,
    self_dtype,
    self_config_is_decoder: bool,
    dtype=None,
) -> Tensor:
    """
    See ModuleUtilsMixin

    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if dtype is None:
        dtype = self_dtype

    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        if self_config_is_decoder:
            extended_attention_mask = (
                ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask
                )
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    extended_attention_mask = extended_attention_mask.to(
        dtype=dtype
    )  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


def _invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).

    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
    # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
    # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
    # /transformer/transformer_layers.py#L270
    # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
    # encoder_extended_attention_mask.transpose(-1, -2))
    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
        dtype=self.dtype
    )  # fp16 compatibility
    encoder_extended_attention_mask = (
        1.0 - encoder_extended_attention_mask
    ) * torch.finfo(self.dtype).min

    return encoder_extended_attention_mask


def BertModel_get_extended_encoder_attention_mask(
    encoder_hidden_states: Tensor, encoder_attention_mask: Tensor, device
) -> Tensor:
    (
        encoder_batch_size,
        encoder_sequence_length,
        _,
    ) = encoder_hidden_states.size()
    encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
    if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
    encoder_extended_attention_mask = _invert_attention_mask(encoder_attention_mask)
    return encoder_extended_attention_mask


def BertLMHeadModel_compute_loss(
    self_loss_fct, prediction_scores, self_config_vocab_size
):
    shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    # TODO: are you sure this code works? labels seem to be undefined
    labels = labels[:, 1:].contiguous()
    # loss_fct = CrossEntropyLoss()
    lm_loss = self_loss_fct(
        shifted_prediction_scores.view(-1, self_config_vocab_size),
        labels.view(-1),
    )
    return lm_loss


bert_leaf_funcs = [
    (
        "BertSelfAttention_add_relative_position_embedding",
        BertSelfAttention_add_relative_position_embedding,
    ),
    (
        "BertSelfAttention_get_new_context_layer_shape",
        BertSelfAttention_get_new_context_layer_shape,
    ),
    (
        "BertModel_get_input_shape_batch_size_seq_length_and_device",
        BertModel_get_input_shape_batch_size_seq_length_and_device,
    ),
    (
        "BertModel_create_default_attention_mask",
        BertModel_create_default_attention_mask,
    ),
    (
        "BertModel_create_default_token_type_ids",
        BertModel_create_default_token_type_ids,
    ),
    ("BertModel_get_extended_attention_mask", BertModel_get_extended_attention_mask),
    (
        "BertModel_get_extended_encoder_attention_mask",
        BertModel_get_extended_encoder_attention_mask,
    ),
    ("BertLMHeadModel_compute_loss", BertLMHeadModel_compute_loss),
]
