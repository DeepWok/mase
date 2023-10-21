from .language_modeling import (
    LanguageModelingDatasetC4,
    LanguageModelingDatasetPTB,
    LanguageModelingDatasetWikitext2,
    LanguageModelingDatasetWikitext103,
    LanguageModelingDatasetScienceQA,
)
from .sentiment_analysis import (
    SentimentalAnalysisDatasetSST2,
    SentimentalAnalysisDatasetCoLa,
)

from .text_entailment import (
    TextEntailmentDatasetBoolQ,
    TextEntailmentDatasetMNLI,
    TextEntailmentDatasetQNLI,
    TextEntailmentDatasetRTE,
    TextEntailmentDatasetQQP,
    TextEntailmentDatasetMRPC,
    TextEntailmentDatasetSTSB,
)
from .translation import (
    TranslationDatasetIWSLT2017_DE_EN,
    TranslationDatasetIWSLT2017_EN_CH,
    TranslationDatasetIWSLT2017_EN_DE,
    TranslationDatasetIWSLT2017_EN_FR,
    TranslationDatasetWMT16_RO_EN,
    TranslationDatasetWMT19_DE_EN,
    TranslationDatasetWMT19_ZH_EN,
)


def get_nlp_dataset(
    name: str,
    split: str,
    tokenizer,
    max_token_len: int,
    num_workers: int,
    load_from_cache_file: bool = True,
    auto_setup: bool = True,
):
    ori_split = split
    assert split in [
        "train",
        "validation",
        "test",
        "pred",
    ], f"Unknown split {split}, should be one of ['train', 'validation', 'test', 'pred']"

    match name:
        case "sst2":
            dataset_cls = SentimentalAnalysisDatasetSST2
        case "cola":
            dataset_cls = SentimentalAnalysisDatasetCoLa
        case "mnli":
            if split == "validation":
                split = "validation_matched"
            dataset_cls = TextEntailmentDatasetMNLI
        case "qnli":
            dataset_cls = TextEntailmentDatasetQNLI
        case "rte":
            dataset_cls = TextEntailmentDatasetRTE
        case "stsb":
            dataset_cls = TextEntailmentDatasetSTSB
        case "qqp":
            dataset_cls = TextEntailmentDatasetQQP
        case "mrpc":
            dataset_cls = TextEntailmentDatasetMRPC
        case "boolq":
            dataset_cls = TextEntailmentDatasetBoolQ
        case "wikitext2":
            dataset_cls = LanguageModelingDatasetWikitext2
        case "wikitext103":
            dataset_cls = LanguageModelingDatasetWikitext103
        case "c4":
            dataset_cls = LanguageModelingDatasetC4
        case "ptb":
            dataset_cls = LanguageModelingDatasetPTB
        case "scienceqa":
            dataset_cls = LanguageModelingDatasetScienceQA
        case "wmt19_de_en":
            dataset_cls = TranslationDatasetWMT19_DE_EN
        case "wmt19_zh_en":
            dataset_cls = TranslationDatasetWMT19_ZH_EN
        case "iwslt2017_en_de":
            dataset_cls = TranslationDatasetIWSLT2017_EN_DE
        case "iwslt2017_de_en":
            dataset_cls = TranslationDatasetIWSLT2017_DE_EN
        case "iwslt2017_en_fr":
            dataset_cls = TranslationDatasetIWSLT2017_EN_FR
        case "iwslt2017_en_ch":
            dataset_cls = TranslationDatasetIWSLT2017_EN_CH
        case "wmt16_ro_en":
            dataset_cls = TranslationDatasetWMT16_RO_EN
        case _:
            raise ValueError(f"Unknown dataset {name}")

    if ori_split == "test" and not dataset_cls.info.test_split_available:
        return None

    if ori_split == "pred" and not dataset_cls.info.pred_split_available:
        return None

    if ori_split == "pred" and dataset_cls.info.pred_split_available:
        split = "test"

    dataset = dataset_cls(
        split,
        tokenizer,
        max_token_len,
        num_workers,
        load_from_cache_file,
        auto_setup,
    )
    return dataset


NLP_DATASET_MAPPING = {
    # CLS dataset
    "sst2": SentimentalAnalysisDatasetSST2,
    "cola": SentimentalAnalysisDatasetCoLa,
    "mnli": TextEntailmentDatasetMNLI,
    "qnli": TextEntailmentDatasetQNLI,
    "rte": TextEntailmentDatasetRTE,
    "qqp": TextEntailmentDatasetQQP,
    "mrpc": TextEntailmentDatasetMRPC,
    "stsb": TextEntailmentDatasetSTSB,
    "boolq": TextEntailmentDatasetBoolQ,
    # Translation dataset
    "wmt19_de_en": TranslationDatasetWMT19_DE_EN,
    "wmt19_zh_en": TranslationDatasetWMT19_ZH_EN,
    "iwslt2017_en_de": TranslationDatasetIWSLT2017_EN_DE,
    "iwslt2017_de_en": TranslationDatasetIWSLT2017_DE_EN,
    "iwslt2017_en_fr": TranslationDatasetIWSLT2017_EN_FR,
    "iwslt2017_en_ch": TranslationDatasetIWSLT2017_EN_CH,
    "wmt16_ro_en": TranslationDatasetWMT16_RO_EN,
    # LM dataset
    "wikitext2": LanguageModelingDatasetWikitext2,
    "wikitext103": LanguageModelingDatasetWikitext103,
    "c4": LanguageModelingDatasetC4,
    "ptb": LanguageModelingDatasetPTB,
    "scienceqa": LanguageModelingDatasetScienceQA,
}


def get_nlp_dataset_cls(name: str):
    assert name in NLP_DATASET_MAPPING, f"Unknown dataset {name}"
    return NLP_DATASET_MAPPING[name]
