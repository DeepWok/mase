from chop.passes.graph.transforms.quantize.quantized_modules.attention_head import (
    BertSelfAttentionHeadInteger,
)
from transformers import AutoConfig
import torch

cf = AutoConfig.from_pretrained("bert-base-uncased")
head = BertSelfAttentionHeadInteger(
    cf, q_config={"data_in_width": 8, "data_in_frac_width": 4}
)

inputs = {
    "query_layer": torch.randn((20, 64)),
    "key_layer": torch.randn((20, 64)),
    "value_layer": torch.randn((20, 64)),
}

_ = head(**inputs)
