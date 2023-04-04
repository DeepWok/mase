from ...session.plt_wrapper import get_model_wrapper
from ...session.plt_wrapper.nlp.classification import NLPClassificationModelWrapper
from ...session.plt_wrapper.nlp.lm import NLPLanguageModelingModelWrapper
from ...session.plt_wrapper.nlp.translation import NLPTranslationModelWrapper
from ...session.plt_wrapper.vision import VisionModelWrapper


# FIXME
# !: This should be consistent with mase-tools/software/machop/graph/dummy_inputs.py
# !: Because dummy_inputs is used to generate fx.Graph
# !: The reason why we use both get_inputs_args and get_dummy_inputs is that
# !: Interpreter.run only supports args, which is offered by get_input_args
# !: But we need get_dummy_inputs to offer both args and kwargs to generate graph
def get_input_args(model_name, task, data_module):
    data_module.prepare_data()
    data_module.setup()

    wrapper_cls = get_model_wrapper(model_name, task)

    if wrapper_cls == VisionModelWrapper:
        batch_x, _ = next(iter(data_module.train_dataloader()))
        input_args = [batch_x[[0], ...]]
    elif wrapper_cls == NLPClassificationModelWrapper:
        batch = next(iter(data_module.train_dataloader()))
        input_args = [
            batch["input_ids"][[0], ...],
            batch["attention_mask"][[0], ...],
        ]
    elif wrapper_cls == NLPLanguageModelingModelWrapper:
        batch = next(iter(data_module.train_dataloader()))
        input_args = [
            batch["input_ids"][[0], ...],
            batch["attention_mask"][[0], ...],
            batch["labels"][[0], ...],
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
