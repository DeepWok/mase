from .resnet import (ResNet18, ResNet50, ResNet18ImageNet, ResNet50ImageNet)
from .nlp_models import get_nlp_model
from .toy import get_toynet
from .manual.toy_manual import get_toymanualnet

model_map = {
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'resnet18-imagenet': ResNet18ImageNet,
    'resnet50-imagenet': ResNet50ImageNet,
    # this is a normal toynet written purely with pytorch ops
    'toy': get_toynet,
    # this is a toynet with our custom ops
    'toy_manual': get_toymanualnet,
    'roberta-base': get_nlp_model,
    'roberta-large': get_nlp_model,
    'facebook/opt-350m': get_nlp_model,
}

# this is a list of models that are written purely with custom ops
# this is necessary for cli to find an opportunity to pass the modify config...
manual_models = ['toy_manual']

vision_models = [
    'toy', 'toy_manual', 'resnet18', 'resnet50', 'resnet18-imagenet',
    'resnet50-imagenet'
]

nlp_models = ['roberta-base', 'roberta-large', 'facebook/opt-350m']
