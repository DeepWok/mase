from .sentiment_analysis import SentAnalDatasetSST2
from .text_entailment import TextEntailDatasetMNLI, TextEntailDatasetQNLI
from .translation import (TranslationDatasetIWSLT2017_EN_DE,
                          TranslationDatasetOPUS_EN_FR,
                          TranslationDatasetMulti30k,
                          TranslationDatasetWMT16_RO_EN)
from .language_modeling import LanguageModelingDatasetWikitext2, LanguageModelingDatasetWikiText103

dataset_factory = {
    "sst2": SentAnalDatasetSST2,
    "mnli": TextEntailDatasetMNLI,
    "qnli": TextEntailDatasetQNLI,
    "iwslt2017_en_de": TranslationDatasetIWSLT2017_EN_DE,
    'opus_en_fr': TranslationDatasetOPUS_EN_FR,
    'multi30k': TranslationDatasetMulti30k,
    'wmt16_ro_en': TranslationDatasetWMT16_RO_EN,
    'wikitext2': LanguageModelingDatasetWikitext2,
    'wikitext103': LanguageModelingDatasetWikiText103,
}