from functools import partial

from .manual.toy_manual import get_toymanualnet
from .nlp_models import get_nlp_model
from .patched_nlp_models import get_patched_nlp_model
from .toy import get_toy_tiny, get_toynet
from .vision import (
    cswin_64_small,
    cswin_64_tiny,
    cswin_96_base,
    cswin_144_large,
    deit_base_patch16_224,
    deit_small_patch16_224,
    deit_tiny_patch16_224,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
    get_resnet18,
    get_resnet34,
    get_resnet50,
    get_resnet101,
    mobilenetv3_large,
    mobilenetv3_small,
    pvt_large,
    pvt_medium,
    pvt_small,
    pvt_tiny,
    pvt_v2_b0,
    pvt_v2_b1,
    pvt_v2_b2,
    pvt_v2_b3,
    pvt_v2_b4,
    pvt_v2_b5,
    wideresnet28_cifar,
)

_built_in_vision_model_map = {
    # resnet
    "resnet18": get_resnet18,
    "resnet34": get_resnet34,
    "resnet50": get_resnet50,
    "resnet101": get_resnet101,
    # wide resnet
    "wideresnet28_cifar": wideresnet28_cifar,
    # mobilenet
    "mobilenetv3_small": mobilenetv3_small,
    "mobilenetv3_large": mobilenetv3_large,
    # efficient net
    "efficientnet_v2_s": efficientnet_v2_s,
    "efficientnet_v2_m": efficientnet_v2_m,
    "efficientnet_v2_l": efficientnet_v2_l,
    # pvt family, originally designed for imagenet
    "pvt_tiny": pvt_tiny,
    "pvt_small": pvt_small,
    "pvt_medium": pvt_medium,
    "pvt_large": pvt_large,
    # pvt v2
    "pvt_v2_b0": pvt_v2_b0,
    "pvt_v2_b1": pvt_v2_b1,
    "pvt_v2_b2": pvt_v2_b2,
    "pvt_v2_b3": pvt_v2_b3,
    "pvt_v2_b4": pvt_v2_b4,
    "pvt_v2_b5": pvt_v2_b5,
    # deit family
    "deit_tiny_224": deit_tiny_patch16_224,
    "deit_small_224": deit_small_patch16_224,
    "deit_base_224": deit_base_patch16_224,
    # cswin family
    "cswin_64_tiny": cswin_64_tiny,
    "cswin_64_small": cswin_64_small,
    "cswin_96_base": cswin_96_base,
    "cswin_144_large": cswin_144_large,
    # this is a normal toynet written purely with pytorch ops
    "toy": get_toynet,
    "toy-tiny": get_toy_tiny,
}

_patched_vision_model_map = {}

# this is a list of models that are written purely with custom ops
# this is necessary for cli to find an opportunity to pass the modify config...

_manual_vision_model_map = {
    # this is a toynet with our custom ops
    "toy_manual": get_toymanualnet,
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
    "facebook/opt-125m@patched": get_patched_nlp_model,
    "facebook/opt-350m@patched": get_patched_nlp_model,
    "facebook/opt-1.3b@patched": get_patched_nlp_model,
    "facebook/opt-2.7b@patched": get_patched_nlp_model,
    "facebook/opt-6.7b@patched": get_patched_nlp_model,
    "facebook/opt-13b@patched": get_patched_nlp_model,
    "facebook/opt-30b@patched": get_patched_nlp_model,
    "facebook/opt-66b@patched": get_patched_nlp_model,
}

_manual_nlp_module_map = {}

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
# model_map = {
#     "resnet18": get_resnet18,
#     "resnet34": get_resnet34,
#     "resnet50": get_resnet50,
#     # "resnet18-imagenet": get_resnet18_imagenet,
#     # "resnet50-imagenet": get_resnet50_imagenet,
#     # # !: test
#     # "resnet18-tv-imagenet": get_resnet18_tv_imagenet,
#     # !: test
#     "wideresnet28_cifar": wideresnet28_cifar,
#     "mobilenetv3_small": mobilenetv3_small,
#     "mobilenetv3_large": mobilenetv3_large,
#     "efficientnet_v2_s": efficientnet_v2_s,
#     "efficientnet_v2_m": efficientnet_v2_m,
#     "efficientnet_v2_l": efficientnet_v2_l,
#     # pvt family, originally designed for imagenet
#     "pvt_tiny": pvt_tiny,
#     "pvt_small": pvt_small,
#     "pvt_medium": pvt_medium,
#     "pvt_large": pvt_large,
#     # pvt v2
#     "pvt_v2_b0": pvt_v2_b0,
#     "pvt_v2_b1": pvt_v2_b1,
#     "pvt_v2_b2": pvt_v2_b2,
#     "pvt_v2_b3": pvt_v2_b3,
#     "pvt_v2_b4": pvt_v2_b4,
#     "pvt_v2_b5": pvt_v2_b5,
#     # deit family
#     "deit_tiny_224": deit_tiny_patch16_224,
#     "deit_small_224": deit_small_patch16_224,
#     "deit_base_224": deit_base_patch16_224,
#     # cswin family
#     "cswin_64_tiny": cswin_64_tiny,
#     "cswin_64_small": cswin_64_small,
#     "cswin_96_base": cswin_96_base,
#     "cswin_144_large": cswin_144_large,
#     # this is a normal toynet written purely with pytorch ops
#     "toy": get_toynet,
#     "toy-tiny": get_toy_tiny,
#     # this is a toynet with our custom ops
#     "toy_manual": get_toymanualnet,
#     # language models
#     "bert-base-uncased": get_nlp_model,
#     "bert-base-cased": get_nlp_model,
#     "gpt2": get_nlp_model,
#     "roberta-base": get_nlp_model,
#     "roberta-large": get_nlp_model,
#     # opt models
#     "facebook/opt-125m": get_nlp_model,
#     "facebook/opt-350m": get_nlp_model,
#     "facebook/opt-1.3b": get_nlp_model,
#     "facebook/opt-2.7b": get_nlp_model,
#     "facebook/opt-6.7b": get_nlp_model,
#     "facebook/opt-13b": get_nlp_model,
#     "facebook/opt-30b": get_nlp_model,
#     "facebook/opt-66b": get_nlp_model,
#     # gpt neo models
#     "EleutherAI/gpt-neo-125M": get_nlp_model,
#     "EleutherAI/gpt-neo-1.3B": get_nlp_model,
#     "EleutherAI/gpt-neo-2.7B": get_nlp_model,
#     "EleutherAI/gpt-neox-20b": get_nlp_model,
#     # t5 family
#     "t5-small": get_nlp_model,
#     "t5-base": get_nlp_model,
#     "t5-large": get_nlp_model,
#     "google/t5-v1_1-small": get_nlp_model,
#     # ----------------------------------------
#     # Patched NLP models supporting FX.graph 👇
#     # ----------------------------------------
#     "facebook/opt-125m@patched": get_patched_nlp_model,
#     "facebook/opt-350m@patched": get_patched_nlp_model,
#     "facebook/opt-1.3b@patched": get_patched_nlp_model,
#     "facebook/opt-2.7b@patched": get_patched_nlp_model,
#     "facebook/opt-6.7b@patched": get_patched_nlp_model,
#     "facebook/opt-13b@patched": get_patched_nlp_model,
#     "facebook/opt-30b@patched": get_patched_nlp_model,
#     "facebook/opt-66b@patched": get_patched_nlp_model,
# }


# manual_models = list(_vision_model_map)


# vision_models = [
#     # manual models
#     "toy",
#     "toy-tiny",
#     "toy_manual",
#     # built-in models
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     # "resnet18-imagenet",
#     # "resnet50-imagenet",
#     # # !: test
#     # "resnet18-tv-imagenet",
#     # !: test
#     "mobilenetv3_small",
#     "mobilenetv3_large",
#     "efficientnet_v2_s",
#     "efficientnet_v2_m",
#     "efficientnet_v2_l",
#     "pvt_tiny",
#     "pvt_small",
#     "pvt_medium",
#     "pvt_large",
#     "pvt_v2_b0",
#     "pvt_v2_b1",
#     "pvt_v2_b2",
#     "pvt_v2_b3",
#     "pvt_v2_b4",
#     "pvt_v2_b5",
#     # deit family
#     "deit_tiny_224",
#     "deit_small_224",
#     "deit_base_224",
#     # cswin family
#     "cswin_64_tiny",
#     "cswin_64_small",
#     "cswin_96_base",
#     "cswin_144_large",
# ]

# nlp_models = [
#     "bert-base-uncased",
#     "bert-base-cased",
#     "gpt2",
#     "roberta-base",
#     "roberta-large",
#     "facebook/opt-125m",
#     "facebook/opt-350m",
#     "facebook/opt-1.3b",
#     "facebook/opt-2.7b",
#     "facebook/opt-6.7b",
#     "facebook/opt-13b",
#     "facebook/opt-66b",
#     "facebook/opt-30b",
#     "EleutherAI/gpt-neo-125M",
#     "EleutherAI/gpt-neo-1.3B",
#     "EleutherAI/gpt-neo-2.7B",
#     "EleutherAI/gpt-neox-20b",
#     "t5-small",
#     "t5-base",
#     "t5-large",
#     "google/t5-v1_1-small",
#     # ----------------------------------------
#     # Patched NLP models supporting FX.graph 👇
#     # ----------------------------------------
#     "facebook/opt-125m@patched",
#     "facebook/opt-350m@patched",
#     "facebook/opt-1.3b@patched",
#     "facebook/opt-2.7b@patched",
#     "facebook/opt-6.7b@patched",
#     "facebook/opt-13b@patched",
#     "facebook/opt-30b@patched",
#     "facebook/opt-66b@patched",
# ]

# patched_nlp_models = [
#     "facebook/opt-125m@patched",
#     "facebook/opt-350m@patched",
#     "facebook/opt-1.3b@patched",
#     "facebook/opt-2.7b@patched",
#     "facebook/opt-6.7b@patched",
#     "facebook/opt-13b@patched",
#     "facebook/opt-30b@patched",
#     "facebook/opt-66b@patched",
# ]
