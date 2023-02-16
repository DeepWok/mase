from .resnet import (ResNet18, ResNet50, ResNet18ImageNet, ResNet50ImageNet)
from .nlp_models import get_nlp_model
from .toy import get_toynet

model_map = {
    'resnet18': ResNet18,
    'resnet50': ResNet50,
    'resnet18-imagenet': ResNet18ImageNet,
    'resnet50-imagenet': ResNet50ImageNet,
    'toy': get_toynet,
    'roberta-base': get_nlp_model,
    'roberta-large': get_nlp_model,
}

vision_models = [
    'toy', 'resnet18', 'resnet50', 'resnet18-imagenet', 'resnet50-imagenet'
]

nlp_models = ['roberta-base', 'roberta-large']
