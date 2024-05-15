from transformers.models.bert import BertTokenizer
from .configuration_bert import BertQuantizedConfig
from .modeling_bert import (
    BertQuantizedForMaskedLM,
    BertQuantizedForMultipleChoice,
    BertQuantizedForNextSentencePrediction,
    BertQuantizedForPreTraining,
    BertQuantizedForQuestionAnswering,
    BertQuantizedForSequenceClassification,
    BertQuantizedForTokenClassification,
)
from .quant_config_bert import parse_bert_quantized_config
