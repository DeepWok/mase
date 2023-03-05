from .resnet import (ResNet18, ResNet50, ResNet18ImageNet, ResNet50ImageNet)
from .nlp_models import get_nlp_model
from .toy import get_toynet
from .manual.toy_manual import get_toymanualnet
from functools import partial

model_map = {
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'resnet18-imagenet': ResNet18ImageNet,
    'resnet50-imagenet': ResNet50ImageNet,
    # this is a normal toynet written purely with pytorch ops
    'toy': get_toynet,
    # this is a toynet with our custom ops
    'toy_manual': get_toymanualnet,
    # language models
    'bert-base-uncased': get_nlp_model,
    'gpt2': get_nlp_model,
    'roberta-base': get_nlp_model,
    'roberta-large': get_nlp_model,
    # opt models
    'facebook/opt-125m': get_nlp_model,
    'facebook/opt-350m': get_nlp_model,
    'facebook/opt-1.3b': get_nlp_model,
    'facebook/opt-2.7b': get_nlp_model,
    'facebook/opt-13b': get_nlp_model,
    'facebook/opt-30b': get_nlp_model,
    'facebook/opt-66b': get_nlp_model,
    # gpt neo models
    'EleutherAI/gpt-neo-125M': get_nlp_model,
    'EleutherAI/gpt-neo-1.3B': get_nlp_model,
    'EleutherAI/gpt-neo-2.7B': get_nlp_model,
    'EleutherAI/gpt-neox-20b': get_nlp_model,
}

# this is a list of models that are written purely with custom ops
# this is necessary for cli to find an opportunity to pass the modify config...
manual_models = ['toy_manual']

vision_models = [
    'toy', 'toy_manual', 'resnet18', 'resnet50', 'resnet18-imagenet',
    'resnet50-imagenet'
]

nlp_models = [
    'bert-base-uncased',
    'gpt2',
    'roberta-base',
    'roberta-large',
    'facebook/opt-125m',
    'facebook/opt-350m',
    'facebook/opt-1.3b',
    'facebook/opt-2.7b',
    'facebook/opt-13b',
    'facebook/opt-66b',
    'facebook/opt-30b',
    'EleutherAI/gpt-neo-125M',
    'EleutherAI/gpt-neo-1.3B',
    'EleutherAI/gpt-neo-2.7B',
    'EleutherAI/gpt-neox-20b',
]
