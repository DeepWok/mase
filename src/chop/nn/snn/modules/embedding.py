import torch
from torch import nn
from typing import Optional, Tuple
from torch import Tensor

from chop.nn.snn.modules.neuron import ANN_neuron
from chop.nn.snn.metric.averagemeter import AverageMeter


class EmbeddingZIPTF(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[Tensor] = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            _freeze=_freeze,
            device=device,
            dtype=dtype,
        )
        self.T = 0
        self.shape = None

    def reset(self):
        self.T = 0

    def forward(self, x):
        if self.T == 0:
            output = super().forward(x)
            self.shape = output.shape
            self.T = self.T + 1
            return output
        else:
            return torch.zeros(self.shape, device=x.device)


class RobertaEmbeddingsSA(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config, q_config):
        super().__init__()
        self.ann = True 
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.neuron_1_if = ANN_neuron()
        self.neuron_1_if_meter = AverageMeter()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if(not self.ann):
            max_tp = embeddings.max()
            embeddings = torch.round(embeddings/max_tp*(2**(self.neuron_1_if.timestep)))/(2**(self.neuron_1_if.timestep))*max_tp

        #if(self.ann):
        self.neuron_1_if_meter.update(torch.count_nonzero(embeddings)/torch.numel(embeddings))

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        embeddings = self.neuron_1_if(embeddings)
        return embeddings

    def flops(self):
        flops = torch.numel(self.neuron_1_if.max_act)
        return flops

    def flops_ANN(self):
        energy = ANN_Energy(torch.numel(self.neuron_1_if.max_act),torch.numel(self.neuron_1_if.max_act),self.neuron_1_if_meter.avg)
        return energy
    def flops_SNN(self):
        energy = SNN_Energy(torch.numel(self.neuron_1_if.max_act),torch.numel(self.neuron_1_if.max_act),self.neuron_1_if_meter.avg,1,self.neuron_1_if.timstep_cycle)
        return energy



def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx



