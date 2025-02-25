#!/usr/bin/env python3
# This example converts a simple MLP model to Verilog
import logging
import os
import sys

import torch
import torch.nn as nn

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[5].as_posix())


from chop.passes.module.transforms import mla_transform_pass


# --------------------------------------------------
#   Model specifications
# --------------------------------------------------
class SimpleTransformerEncoder(nn.Module):
    """
    A simple Transformer encoder with a few layers.
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super(SimpleTransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True  # Use batch_first=True for convenience
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # final linear just for demonstration
        self.fc_out = nn.Linear(d_model, 10)
        
    def forward(self, x):
        """
        x is expected to be of shape [batch_size, seq_len, d_model]
        """
        enc_output = self.transformer_encoder(x)  # (B, S, D)
        output = self.fc_out(enc_output[:, 0, :]) # classify just first token
        return output


def test_mla_transform_pass():
    mlp = SimpleTransformerEncoder()
    # Sanity check and report
    # mg = verify_common_metadata_analysis_pass(mg)
    pass_args = {
        "by": "name",
        "fc1": {
            "config": {
                "name": "integer",
                "data_in_width": 8,
                "data_in_frac_width": 4,
                "weight_width": 8,
                "weight_frac_width": 4,
                "bias_width": 8,
                "bias_frac_width": 4,
            }
        },
    }
    mla_transform_pass(mlp, pass_args)


test_mla_transform_pass()
