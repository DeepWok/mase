from dataclasses import dataclass
from enum import Enum


class ModelSource(Enum):
    """
    The source of the model, must be one of the following:
    - HF: HuggingFace
    - MANUAL: manually implemented
    - PATCHED: patched HuggingFace
    - TOY: toy model for testing and debugging
    - PHYSICAL: model that perform classification using physical data point vectors
    - NERF: model that estimates neural radiance field (NeRF) of a 3D scene
    """

    HF_TRANSFORMERS = "hf_transformers"
    MANUAL = "manual"
    PATCHED = "patched"
    TOY = "toy"
    TORCHVISION = "torchvision"
    VISION_OTHERS = "vision_others"
    PHYSICAL = "physical"
    NERF = "nerf"
    NERF_VISION = "nerfvision"


class ModelTaskType(Enum):
    """
    The task type of the model, must be one of the following:
    - NLP: natural language processing
    - VISION: computer vision
    - PHYSICAL: categorize data points into predefined classes based on their features or attributes
    - NERF: estimate neural radiance field (NeRF) of a 3D scene
    """

    NLP = "nlp"
    VISION = "vision"
    PHYSICAL = "physical"
    NERF = "nerf"
    NERF_VISION = "nerfvision"


@dataclass
class MaseModelInfo:
    """
    The model info for MASE.
    """

    # model name
    name: str

    model_source: ModelSource
    task_type: ModelTaskType

    # Vision models
    image_classification: bool = False

    # Physical models
    physical_data_point_classification: bool = False

    # NLP models
    sequence_classification: bool = False
    seq2seqLM: bool = False
    causal_LM: bool = False

    # Manual models
    is_quantized: bool = False
    is_lora: bool = False
    is_sparse: bool = False

    # Torch FX
    is_fx_traceable: bool = False

    def __post_init__(self):
        self.model_source = (
            ModelSource(self.model_source)
            if isinstance(self.model_source, str)
            else self.model_source
        )
        self.task_type = (
            ModelTaskType(self.task_type)
            if isinstance(self.task_type, str)
            else self.task_type
        )

        # NLP models
        if self.task_type == ModelTaskType.NLP:
            assert self.sequence_classification + self.seq2seqLM + self.causal_LM >= 1

        # Vision models
        if self.task_type == ModelTaskType.VISION:
            assert self.image_classification, "Must be an image classification model"

        # Classification models
        if self.task_type == ModelTaskType.PHYSICAL:
            assert (
                self.physical_data_point_classification
            ), "Must be an physical data point classification model"

        if self.task_type == ModelTaskType.NERF:
            # TODO:
            pass

        if self.task_type == ModelTaskType.NERF_VISION:
            # TODO:
            pass

        # manual models
        assert self.is_quantized + self.is_lora + self.is_sparse <= 1
        if self.is_quantized or self.is_lora or self.is_sparse:
            assert self.model_source == ModelSource.MANUAL, "Must be a manual model"

    @property
    def is_nlp_model(self):
        return self.task_type == ModelTaskType.NLP

    @property
    def is_vision_model(self):
        return self.task_type == ModelTaskType.VISION

    @property
    def is_physical_model(self):
        return self.task_type == ModelTaskType.PHYSICAL

    @property
    def is_nerf_model(self):
        return self.task_type == ModelTaskType.NERF

    @property
    def is_nerf_vision_model(self):
        return self.task_type == ModelTaskType.NERF_VISION


class ModelFactory:
    _model_info_dict: dict = {}
    _model_cls_dict: dict = {}
    _checkpoint_getter_dict: dict = {}

    def __init__(self, model):
        self.model = model


def register_mase_model(
    name: str,
    checkpoints: list[str],
    model_source: ModelSource,
    task_type: ModelTaskType,
    image_classification: bool = False,
    physical_data_point_classification: bool = False,
    sequence_classification: bool = False,
    seq2seqLM: bool = False,
    causal_LM: bool = False,
    is_quantized: bool = False,
    is_lora: bool = False,
    is_sparse: bool = False,
    is_fx_traceable: bool = False,
):
    """This decorator registers models as ModelFactory class attributes which can be used globally."""

    def decorator(cls):

        model_info = MaseModelInfo(
            name=name,
            model_source=model_source,
            task_type=task_type,
            image_classification=image_classification,
            physical_data_point_classification=physical_data_point_classification,
            sequence_classification=sequence_classification,
            seq2seqLM=seq2seqLM,
            causal_LM=causal_LM,
            is_quantized=is_quantized,
            is_lora=is_lora,
            is_sparse=is_sparse,
            is_fx_traceable=is_fx_traceable,
        )

        # Register each checkpoint in the factory with the same model info
        for checkpoint in checkpoints:
            ModelFactory._model_info_dict[checkpoint] = model_info
            ModelFactory._model_cls_dict[checkpoint] = cls

        # Register in the Pytorch module
        cls.mase_model_info = model_info

        return cls

    return decorator


def register_mase_checkpoint(checkpoint: str):
    """This decorator registers checkpoint getters as ModelFactory class attributes which can be used globally."""

    def decorator(cls):
        ModelFactory._checkpoint_getter_dict[checkpoint] = cls
        return cls

    return decorator
