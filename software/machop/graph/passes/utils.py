from ...session.plt_wrapper import get_model_wrapper
from ...session.plt_wrapper.nlp.classification import NLPClassificationModelWrapper
from ...session.plt_wrapper.nlp.lm import NLPLanguageModelingModelWrapper
from ...session.plt_wrapper.nlp.translation import NLPTranslationModelWrapper
from ...session.plt_wrapper.vision import VisionModelWrapper


def get_input_args(model_name, task, data_module):
    data_module.setup()
    data_module.prepare_data()

    wrapper_cls = get_model_wrapper(model_name, task)

    if wrapper_cls == VisionModelWrapper:
        batch_x, _ = next(iter(data_module.train_dataloader()))
        input_args = [batch_x[[0], ...]]
    elif wrapper_cls in [
        NLPClassificationModelWrapper,
        NLPLanguageModelingModelWrapper,
    ]:
        batch = next(iter(data_module.train_dataloader))
        input_args = [
            batch["input_ids"][[0], ...],
            batch["attention_mask"][[0], ...],
            None,
        ]
    else:
        # NLPTranslationModelWrapper
        batch = next(iter(data_module.train_dataloader))
        input_args = [
            batch["input_ids"][[0], ...],
            batch["attention_mask"][[0], ...],
            batch["decoder_input_ids"][[0], ...],
            batch["decoder_attention_mask"][[0], ...],
        ]
    return input_args
