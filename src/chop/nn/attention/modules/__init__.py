from .mla import MLA
from .mgqa import MGQALayers

attention_module_map = {
    "attention_latent": MLA,
    "attention_gpa": MGQALayers
}