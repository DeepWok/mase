import torch
from torch import nn
from typing import Optional, Tuple
from torch import Tensor


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
