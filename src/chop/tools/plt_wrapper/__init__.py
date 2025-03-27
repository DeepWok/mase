from .vision import VisionModelWrapper
from .physical import JetSubstructureModelWrapper
from .nerf import NeRFModelWrapper
from .nlp import (
    NLPClassificationModelWrapper,
    NLPLanguageModelingModelWrapper,
    NLPTranslationModelWrapper,
)
from .vision.ultralytics_detection import UltralyticsDetectionWrapper
from .vision.ultralytics_segmentation import UltralyticsSegmentationWrapper


def get_model_wrapper(model_info, task: str):
    if model_info.is_physical_model:
        return JetSubstructureModelWrapper
    elif model_info.is_nerf_model:
        return NeRFModelWrapper
    elif model_info.is_vision_model:
        match task:
            case "detection":
                return UltralyticsDetectionWrapper
            case "instance-segmentation":
                return UltralyticsSegmentationWrapper
            case _:
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
