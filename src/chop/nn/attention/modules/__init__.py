from .mla import MLA
from .mgqa import MGQALayers
from .llama import LlamaForCausalLM

attention_module_map = {"attention_latent": MLA, "attention_gpa": MGQALayers}
