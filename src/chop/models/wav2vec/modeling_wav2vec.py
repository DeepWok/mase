import logging
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from chop.models.utils import register_mase_model, register_mase_checkpoint

logger = logging.getLogger(__name__)

@register_mase_model(
    name="wav2vec",
    checkpoints=[
        "wav2vec2-base",
        "wav2vec2-base-960h",
        "wav2vec2-large",
        "wav2vec2-large-960h",
        "wav2vec2-large-xlsr"
    ],
    model_source="hf_transformers",
    task_type="speech",
    is_fx_traceable=True,
)

class Wav2VecModelWrapper(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.model = Wav2Vec2Model(config)
        
    def forward(self, input_values, **kwargs):
        return self.model(input_values, **kwargs)

def _get_wav2vec_model(model_size: str, pretrained: bool = False, **kwargs):
    """Helper function to get Wav2Vec models"""
    if pretrained:
        model_id = f"facebook/wav2vec2-{model_size}"
        model = Wav2Vec2Model.from_pretrained(model_id)
        logger.info(f"Loaded pretrained Wav2Vec2 {model_size} model")
    else:
        config = Wav2Vec2Config.from_pretrained(f"facebook/wav2vec2-{model_size}")
        model = Wav2VecModelWrapper(config)
        logger.info(f"Initialized Wav2Vec2 {model_size} model from config")
    
    return model

@register_mase_checkpoint("wav2vec2-base")
def get_wav2vec_base(pretrained: bool = False, **kwargs):
    return _get_wav2vec_model("base", pretrained, **kwargs)

@register_mase_checkpoint("wav2vec2-base-960h")
def get_wav2vec_base_960h(pretrained: bool = False, **kwargs):
    return _get_wav2vec_model("base", pretrained, **kwargs)

@register_mase_checkpoint("wav2vec2-large")
def get_wav2vec_large(pretrained: bool = False, **kwargs):
    return _get_wav2vec_model("large", pretrained, **kwargs)

@register_mase_checkpoint("wav2vec2-large-960h")
def get_wav2vec_large_960h(pretrained: bool = False, **kwargs):
    return _get_wav2vec_model("large-960h", pretrained, **kwargs)

@register_mase_checkpoint("wav2vec2-large-xlsr")
def get_wav2vec_large_xlsr(pretrained: bool = False, **kwargs):
    return _get_wav2vec_model("large-xlsr-53", pretrained, **kwargs) 