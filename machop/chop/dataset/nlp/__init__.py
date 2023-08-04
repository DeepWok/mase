from .language_modeling import (
    LanguageModelingDatasetC4,
    LanguageModelingDatasetPTB,
    LanguageModelingDatasetWikitext2,
    LanguageModelingDatasetWikiText103,
    LanguageModelingDatasetScienceQA,
)
from .sentiment_analysis import SentAnalDatasetSST2
from .text_entailment import (
    TextEntailDatasetBoolQ,
    TextEntailDatasetMNLI,
    TextEntailDatasetQNLI,
)
from .translation import (
    TranslationDatasetIWSLT2017_DE_EN,
    TranslationDatasetIWSLT2017_EN_CH,
    TranslationDatasetIWSLT2017_EN_DE,
    TranslationDatasetIWSLT2017_EN_FR,
    TranslationDatasetMulti30k,
    TranslationDatasetWMT16_RO_EN,
    TranslationDatasetWMT19_DE_EN,
    TranslationDatasetWMT19_ZH_EN,
)

dataset_factory = {
    # CLS dataset
    "sst2": SentAnalDatasetSST2,
    "mnli": TextEntailDatasetMNLI,
    "qnli": TextEntailDatasetQNLI,
    "boolq": TextEntailDatasetBoolQ,
    # Translation dataset
    # 'opus_en_fr': TranslationDatasetOPUS_EN_FR,
    "multi30k": TranslationDatasetMulti30k,
    "wmt19_de_en": TranslationDatasetWMT19_DE_EN,
    "wmt19_zh_en": TranslationDatasetWMT19_ZH_EN,
    "iwslt2017_en_de": TranslationDatasetIWSLT2017_EN_DE,
    "iwslt2017_de_en": TranslationDatasetIWSLT2017_DE_EN,
    "iwslt2017_en_fr": TranslationDatasetIWSLT2017_EN_FR,
    "iwslt2017_en_ch": TranslationDatasetIWSLT2017_EN_CH,
    "wmt16_ro_en": TranslationDatasetWMT16_RO_EN,
    # LM dataset
    "wikitext2": LanguageModelingDatasetWikitext2,
    "wikitext103": LanguageModelingDatasetWikiText103,
    "c4": LanguageModelingDatasetC4,
    "ptb": LanguageModelingDatasetPTB,
    "scienceqa": LanguageModelingDatasetScienceQA,
}
