from .vision import (
    get_resnet18,
    get_resnet50,
    get_resnet101,
    get_resnet18_imagenet,
    get_resnet50_imagenet,
)
from .vision import pvt_tiny, pvt_small, pvt_medium, pvt_large
from .vision import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
from .vision import wideresnet28_cifar
from .vision import cswin_64_tiny, cswin_64_small, cswin_96_base, cswin_144_large
from .vision import deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224

from .nlp_models import get_nlp_model
from .toy import get_toynet
from .manual.toy_manual import get_toymanualnet
from functools import partial

model_map = {
    "resnet18": get_resnet18,
    "resnet50": get_resnet50,
    "resnet18-imagenet": get_resnet18_imagenet,
    "resnet50-imagenet": get_resnet50_imagenet,
    "wideresnet28_cifar": wideresnet28_cifar,
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
    # this is a toynet with our custom ops
    "toy_manual": get_toymanualnet,
    # language models
    "bert-base-uncased": get_nlp_model,
    "gpt2": get_nlp_model,
    "roberta-base": get_nlp_model,
    "roberta-large": get_nlp_model,
    # opt models
    "facebook/opt-125m": get_nlp_model,
    "facebook/opt-350m": get_nlp_model,
    "facebook/opt-1.3b": get_nlp_model,
    "facebook/opt-2.7b": get_nlp_model,
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

# this is a list of models that are written purely with custom ops
# this is necessary for cli to find an opportunity to pass the modify config...
manual_models = ["toy_manual"]

vision_models = [
    "toy",
    "toy_manual",
    "resnet18",
    "resnet50",
    "resnet18-imagenet",
    "resnet50-imagenet",
    "pvt_tiny",
    "pvt_small",
    "pvt_medium",
    "pvt_large",
    "pvt_v2_b0",
    "pvt_v2_b1",
    "pvt_v2_b2",
    "pvt_v2_b3",
    "pvt_v2_b4",
    "pvt_v2_b5",
    # deit family
    "deit_tiny_224",
    "deit_small_224",
    "deit_base_224",
    # cswin family
    "cswin_64_tiny",
    "cswin_64_small",
    "cswin_96_base",
    "cswin_144_large",
]

nlp_models = [
    "bert-base-uncased",
    "gpt2",
    "roberta-base",
    "roberta-large",
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-13b",
    "facebook/opt-66b",
    "facebook/opt-30b",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neox-20b",
    "t5-small",
    "t5-base",
    "t5-large",
    "google/t5-v1_1-small",
]
