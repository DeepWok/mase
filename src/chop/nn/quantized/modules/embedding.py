from functools import partial

import torch
from torch import Tensor, nn

from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer


class EmbeddingMXFP(nn.Embedding):
    """MXFP-quantized Embedding. Weight is quantized at forward time."""

    def __init__(self, num_embeddings, embedding_dim, config=None, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.config = config or {}
        self.bypass = self.config.get("bypass", False)

        if not self.bypass:
            self.w_quantizer = partial(
                mxfp_quantizer,
                block_size=self.config["weight_block_size"],
                element_exp_bits=self.config["weight_exponent_width"],
                element_frac_bits=self.config["weight_frac_width"],
                block_dim=1,
            )

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(input)
        weight = self.w_quantizer(self.weight)
        return torch.nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )


class EmbeddingMXInt(nn.Embedding):
    """MXInt-quantized Embedding. Weight is quantized at forward time."""

    def __init__(self, num_embeddings, embedding_dim, config=None, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.config = config or {}
        self.bypass = self.config.get("bypass", False)

        if not self.bypass:
            self.w_quantizer = partial(
                mxint_quantizer,
                block_size=self.config["weight_block_size"],
                element_bits=self.config["weight_width"],
                block_dim=1,
            )

    @torch.no_grad()
    def forward(self, input: Tensor) -> Tensor:
        if self.bypass:
            return super().forward(input)
        weight = self.w_quantizer(self.weight)
        return torch.nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse,
        )
