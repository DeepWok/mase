from chop.nn.quantized.modules.attention_head import (
    BertSelfAttentionHeadInteger,
)
from transformers import AutoConfig
import torch


def test_quantized_attention_head():
    cf = AutoConfig.from_pretrained("bert-base-uncased")
    head = BertSelfAttentionHeadInteger(cf, q_config={"width": 8, "frac_width": 4})

    inputs = {
        "query_layer": torch.randn((20, 64)),
        "key_layer": torch.randn((20, 64)),
        "value_layer": torch.randn((20, 64)),
    }

    _ = head(**inputs)


if __name__ == "__main__":
    test_quantized_attention_head()
