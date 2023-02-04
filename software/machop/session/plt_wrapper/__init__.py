from .vision import VisionModelWrapper
from .nlp import NLPClassificationModelWrapper
from machop.models import nlp_models, vision_models


def get_model_wrapper(name, task):
    if name in vision_models:
        return VisionModelWrapper
    elif name in nlp_models:
        if task == 'classification':
            return NLPClassificationModelWrapper
        else:
            raise NotImplementedError(f"Task {task} not implemented for NLP models")
    else:
        raise ValueError(f"Model {name} not implemented")
