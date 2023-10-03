from .vision import VisionModelWrapper
from .physical import JetSubstructureModelWrapper
from .nlp import (
    NLPClassificationModelWrapper,
    NLPLanguageModelingModelWrapper,
    NLPTranslationModelWrapper,
)


def get_model_wrapper(model_info, task: str):
    if model_info.is_physical_model:
        return JetSubstructureModelWrapper
    elif model_info.is_vision_model:
        return VisionModelWrapper
    elif model_info.is_nlp_model:
        match task:
            case "classification" | "cls":
                return NLPClassificationModelWrapper
            case "language_modeling" | "lm":
                return NLPLanguageModelingModelWrapper
            case "translation" | "tran":
                return NLPTranslationModelWrapper
            case _:
                raise ValueError(f"Task {task} is not supported for {model_info.name}")
