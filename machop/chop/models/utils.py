from dataclasses import dataclass
from enum import Enum


class ModelSource(Enum):
    """
    The source of the model, must be one of the following:
    - HF: HuggingFace
    - MANUAL: manually implemented
    - PATCHED: patched HuggingFace
    - TOY: toy model for testing and debugging
    """

    HF_TRANSFORMERS = "hf_transformers"
    MANUAL = "manual"
    PATCHED = "patched"
    TOY = "toy"
    TORCHVISION = "torchvision"
    VISION_OTHERS = "vision_others"


class ModelTaskType(Enum):
    """
    The task type of the model, must be one of the following:
    - NLP: natural language processing
    - VISION: computer vision
    """

    NLP = "nlp"
    VISION = "vision"


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
