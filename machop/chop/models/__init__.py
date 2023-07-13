from functools import partial

# TODO: fix patched models
# from .manual.toy_manual import get_toymanualnet
# from .patched_nlp_models import get_patched_nlp_model
from .patched_nlp_models import patched_model_cls_to_required_input_args
from .manual import get_llama_plain, get_opt_plain

from .nlp_models import get_nlp_model
from .toy import get_toy_tiny, get_toynet, get_testmodel
from .toy_custom_fn import get_toyfnnet
from .vision import (
    cswin_64_small,
    cswin_64_tiny,
    cswin_96_base,
    cswin_144_large,
    get_deit_base_patch16_224,
    get_deit_small_patch16_224,
    get_deit_tiny_patch16_224,
    get_efficientnet_b0,
    get_efficientnet_b3,
    get_efficientnet_v2_l,
    get_efficientnet_v2_m,
    get_efficientnet_v2_s,
    get_mobilenet_v2,
    get_mobilenetv3_large,
    get_mobilenetv3_small,
    get_pvt_large,
    get_pvt_medium,
    get_pvt_small,
    get_pvt_tiny,
    get_pvt_v2_b0,
    get_pvt_v2_b1,
    get_pvt_v2_b2,
    get_pvt_v2_b3,
    get_pvt_v2_b4,
    get_pvt_v2_b5,
    get_resnet18,
    get_resnet34,
    get_resnet50,
    get_resnet101,
    get_wide_resnet50_2,
    wideresnet28_cifar,
)

_built_in_vision_model_map = {
    # resnet
    "resnet18": get_resnet18,
    "resnet34": get_resnet34,
    "resnet50": get_resnet50,
    "resnet101": get_resnet101,
    # wide resnet
    "wideresnet50_2": get_wide_resnet50_2,
    "wideresnet28_cifar": wideresnet28_cifar,
    # mobilenet v2
    "mobilenetv2": get_mobilenet_v2,
    # mobilenet v3
    "mobilenetv3_small": get_mobilenetv3_small,
    "mobilenetv3_large": get_mobilenetv3_large,
    # efficient net
    "efficientnet_b0": get_efficientnet_b0,
    "efficientnet_b3": get_efficientnet_b3,
    "efficientnet_v2_s": get_efficientnet_v2_s,
    "efficientnet_v2_m": get_efficientnet_v2_m,
    "efficientnet_v2_l": get_efficientnet_v2_l,
    # pvt family, originally designed for imagenet
    "pvt_tiny": get_pvt_tiny,
    "pvt_small": get_pvt_small,
    "pvt_medium": get_pvt_medium,
    "pvt_large": get_pvt_large,
    # pvt v2
    "pvt_v2_b0": get_pvt_v2_b0,
    "pvt_v2_b1": get_pvt_v2_b1,
    "pvt_v2_b2": get_pvt_v2_b2,
    "pvt_v2_b3": get_pvt_v2_b3,
    "pvt_v2_b4": get_pvt_v2_b4,
    "pvt_v2_b5": get_pvt_v2_b5,
    # deit family
    "deit_tiny_224": get_deit_tiny_patch16_224,
    "deit_small_224": get_deit_small_patch16_224,
    "deit_base_224": get_deit_base_patch16_224,
    # cswin family
    "cswin_64_tiny": cswin_64_tiny,
    "cswin_64_small": cswin_64_small,
    "cswin_96_base": cswin_96_base,
    "cswin_144_large": cswin_144_large,
    # this is a normal toynet written purely with pytorch ops
    "toy": get_toynet,
    "test": get_testmodel,
    "toy-fn": get_toyfnnet,
    "toy-tiny": get_toy_tiny,
}

_patched_vision_model_map = {}

# this is a list of models that are written purely with custom ops
# this is necessary for cli to find an opportunity to pass the modify config...

_manual_vision_model_map = {
    # this is a toynet with our custom ops
    # "toy_manual": get_toymanualnet,
}


_built_in_nlp_model_map = {
    # language models
    "bert-base-uncased": get_nlp_model,
    "bert-base-cased": get_nlp_model,
    "gpt2": get_nlp_model,
    "roberta-base": get_nlp_model,
    "roberta-large": get_nlp_model,
    # opt models
    "facebook/opt-125m": get_nlp_model,
    "facebook/opt-350m": get_nlp_model,
    "facebook/opt-1.3b": get_nlp_model,
    "facebook/opt-2.7b": get_nlp_model,
    "facebook/opt-6.7b": get_nlp_model,
    "facebook/opt-13b": get_nlp_model,
    "facebook/opt-30b": get_nlp_model,
    "facebook/opt-66b": get_nlp_model,
    # gpt neo models
    "EleutherAI/gpt-neo-125M": get_nlp_model,
    "EleutherAI/gpt-neo-1.3B": get_nlp_model,
    "EleutherAI/gpt-neo-2.7B": get_nlp_model,
    "EleutherAI/gpt-neox-20b": get_nlp_model,
    # t5 family
    "t5-small": get_nlp_model,
    "t5-base": get_nlp_model,
    "t5-large": get_nlp_model,
    "google/t5-v1_1-small": get_nlp_model,
}

# ----------------------------------------
# Patched NLP models supporting FX.graph 👇
# ----------------------------------------
_patched_nlp_model_map = {
    # "facebook/opt-125m@patched": get_patched_nlp_model,
    # "facebook/opt-350m@patched": get_patched_nlp_model,
    # "facebook/opt-1.3b@patched": get_patched_nlp_model,
    # "facebook/opt-2.7b@patched": get_patched_nlp_model,
    # "facebook/opt-6.7b@patched": get_patched_nlp_model,
    # "facebook/opt-13b@patched": get_patched_nlp_model,
    # "facebook/opt-30b@patched": get_patched_nlp_model,
    # "facebook/opt-66b@patched": get_patched_nlp_model,
    # "bert-base-uncased@patched": get_patched_nlp_model,
    # "bert-large-uncased@patched": get_patched_nlp_model,
    # "bert-base-cased@patched": get_patched_nlp_model,
    # "bert-large-cased@patched": get_patched_nlp_model,
}

_manual_nlp_module_map = {
    "Cheng98/llama-160m": get_llama_plain,
}

model_map = (
    _built_in_vision_model_map
    | _built_in_nlp_model_map
    | _manual_nlp_module_map
    | _manual_vision_model_map
    | _patched_nlp_model_map
    | _patched_vision_model_map
)

built_in_vision_models = list(_built_in_vision_model_map.keys())
patched_vision_models = list(_patched_vision_model_map.keys())
manual_vision_models = list(_manual_vision_model_map.keys())
vision_models = built_in_vision_models + patched_vision_models + built_in_vision_models

built_in_nlp_models = list(_built_in_nlp_model_map)
patched_nlp_models = list(_patched_nlp_model_map.keys())
manual_nlp_models = list(_manual_nlp_module_map.keys())
nlp_models = built_in_nlp_models + patched_nlp_models + manual_nlp_models
