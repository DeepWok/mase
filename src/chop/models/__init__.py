from os import PathLike
import torch.nn as nn

from .utils import MaseModelInfo, ModelSource, ModelTaskType, ModelFactory

from .bert import *
from .cnv.cnv import *
from .cswin.cswintransformer import *
from .deit.deit import *
from .efficientnet.efficientnet import *
from .jet_substructure.jet_substructure import *
from .lfc.lfc import *
from .llama.modeling_llama import *
from .mistral.modeling_mistral import *
from .mobilenet_v2.mobilenet_v2 import *
from .mobilenet_v3.mobilenetv3 import *
from .nerf.nerf import *
from .opt.modeling_opt import *
from .pvt.pvt import *
from .repvgg.repvgg import *
from .resnet.resnet import *
from .toy import *
from .vgg_cifar.vgg_cifar import *
from .wideresnet.wideresnet import *
from .wav2vec import *
from .yolo import *


def get_model_info(name: str) -> MaseModelInfo:
    return ModelFactory._model_info_dict[name]


def get_model(
    checkpoint: str | PathLike,
    pretrained: bool,
    **kwargs,
):
    model_info = get_model_info(checkpoint)
    try:
        getter = ModelFactory._checkpoint_getter_dict[checkpoint]
    except KeyError:
        raise ValueError(
            f"Model {checkpoint} not supported, please add it to the ModelFactory or pick from the supported models {ModelFactory._checkpoint_getter_dict.keys()}"
        )
    model = getter(pretrained=pretrained, **kwargs)

    return model


# todo: refactor using factory method as above
def get_tokenizer(name: str, checkpoint: str | PathLike = None):
    """
    Get the tokenizer for the model.

    Args:
        name: The name of the model.
        checkpoint: The path to the checkpoint. This is optional for HuggingFace models, but required for manual models.
    """
    model_info = get_model_info(name)
    assert model_info.is_nlp_model, f"Model {name} is not an NLP model"

    match model_info.model_source:
        case ModelSource.HF_TRANSFORMERS:
            tokenizer = get_hf_nlp_model_tokenizer(name, checkpoint)
        case ModelSource.PATCHED:
            tokenizer = get_patched_model_tokenizer(name, checkpoint)
        case _:
            raise ValueError(f"Model source {model_info.model_source} not supported")
    return tokenizer
