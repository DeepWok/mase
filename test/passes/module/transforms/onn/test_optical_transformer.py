import unittest

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig

from chop.passes.module.transforms.onn.transform import (
    OtLinear,
    OtLlamaAttention,
    OtTransformConfig,
    optical_transformer_module_transform_pass,
)


def _calculate_snr(x, noisy_x):
    noise = noisy_x - x

    signal_power = torch.sum(x**2)
    noise_power = torch.sum(noise**2)

    snr = signal_power / noise_power
    snr_db = 10 * torch.log10(snr)
    return snr_db.item()


class TestOnnTransform(unittest.TestCase):
    def test_ot_linear_layer(self):
        linear = torch.nn.Linear(in_features=32, out_features=64)
        onn_cfg = OtTransformConfig.create_default()
        linear_onn = OtLinear.from_linear(linear, **onn_cfg)

        x = torch.randn(2, 32)
        y = linear(x)
        y_onn = linear_onn(x)

        snr = _calculate_snr(y, y_onn)
        assert snr > 23

    def test_ot_llama_attn_layer(self):
        onn_config = OtTransformConfig.create_default()
        onn_config["q_levels"] = 512
        onn_config["q_smooth_factor"] = 0.1
        model_name = "AICrossSim/clm-60m"
        hf_config = LlamaConfig.from_pretrained(model_name)
        batch_size = 1
        seq_len = 16
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads

        attn = LlamaAttention(config=hf_config, layer_idx=0)

        pos_emb = torch.ones(batch_size, seq_len, head_dim)
        x = 3 * torch.randn(batch_size, seq_len, hf_config.hidden_size)

        y, _ = attn(x, (pos_emb, pos_emb), None)
        y: torch.Tensor
        assert y.isfinite().all()

        attn_onn = OtLlamaAttention.from_pretrained(attn, layer_idx=0, **onn_config)
        attn_onn.train()
        for _ in range(3):
            y_onn, _ = attn_onn(x, (pos_emb, pos_emb), None)

            snr = _calculate_snr(y, y_onn)
            print(f"Attn SNR: {snr:.2f} dB")
            assert snr > 1

    def test_optical_transformer_module_transform_pass(self):
        onn_config = OtTransformConfig.create_default()
        onn_config["q_levels"] = 512
        onn_config["q_smooth_factor"] = 0.1
        model_name = "AICrossSim/clm-60m"
        hf_config = LlamaConfig.from_pretrained(model_name)
        batch_size = 1
        seq_len = 16
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads

        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = LlamaAttention(config=hf_config, layer_idx=0)
                self.linear = torch.nn.Linear(
                    in_features=hf_config.hidden_size,
                    out_features=hf_config.hidden_size,
                )

            def forward(self, x, pos_emb):
                attn_output, _ = self.attn(x, (pos_emb, pos_emb), None)
                output = self.linear(attn_output)
                return output, None

        network = Network()
        pos_emb = torch.ones(batch_size, seq_len, head_dim)
        x = 3 * torch.randn(batch_size, seq_len, hf_config.hidden_size)

        y, _ = network(x, pos_emb)
        y: torch.Tensor
        assert y.isfinite().all()

        pass_args = {
            "by": "regex_name",
            "attn": onn_config,
            "linear": onn_config,
            r"attn\.(q|k|v|o)_proj": onn_config,
        }

        network_onn = optical_transformer_module_transform_pass(network, pass_args)
        assert isinstance(network_onn.attn, OtLlamaAttention)
        assert isinstance(network_onn.linear, OtLinear)
        assert isinstance(network_onn.attn.q_proj, OtLinear)
        assert isinstance(network_onn.attn.k_proj, OtLinear)
        assert isinstance(network_onn.attn.v_proj, OtLinear)
        assert isinstance(network_onn.attn.o_proj, OtLinear)

        print(network_onn)
        network_onn.train()
        for _ in range(3):
            y_onn, _ = network_onn(x, pos_emb)
            assert y_onn.isfinite().all()

            snr = _calculate_snr(y, y_onn)
            assert snr > 1
            print(f"Network SNR: {snr:.2f} dB")
