from .sentiment_analysis import SentAnalDatasetSST2
from .text_entailment import TextEntailDatasetMNLI, TextEntailDatasetQNLI
from .translation import (
    TranslationDatasetIWSLT2017_EN_DE, TranslationDatasetIWSLT2017_DE_EN,
    TranslationDatasetIWSLT2017_EN_FR, TranslationDatasetIWSLT2017_EN_CH,
    TranslationDatasetWMT19_DE_EN, TranslationDatasetWMT19_ZH_EN,
    TranslationDatasetMulti30k, TranslationDatasetWMT16_RO_EN)
from .language_modeling import LanguageModelingDatasetWikitext2, LanguageModelingDatasetWikiText103

dataset_factory = {
    "sst2": SentAnalDatasetSST2,
    "mnli": TextEntailDatasetMNLI,
    "qnli": TextEntailDatasetQNLI,
    # 'opus_en_fr': TranslationDatasetOPUS_EN_FR,
    'multi30k': TranslationDatasetMulti30k,
    'wmt19_de_en': TranslationDatasetWMT19_DE_EN,
    'wmt19_zh_en': TranslationDatasetWMT19_ZH_EN,
    "iwslt2017_en_de": TranslationDatasetIWSLT2017_EN_DE,
    "iwslt2017_de_en": TranslationDatasetIWSLT2017_DE_EN,
    "iwslt2017_en_fr": TranslationDatasetIWSLT2017_EN_FR,
    "iwslt2017_en_ch": TranslationDatasetIWSLT2017_EN_CH,
    'wmt16_ro_en': TranslationDatasetWMT16_RO_EN,
    'wikitext2': LanguageModelingDatasetWikitext2,
    'wikitext103': LanguageModelingDatasetWikiText103,
}
