from functools import partial

from ..dataset import MyDataModule
from ..session.plt_wrapper import get_model_wrapper
from ..session.plt_wrapper.nlp.classification import NLPClassificationModelWrapper
from ..session.plt_wrapper.nlp.lm import NLPLanguageModelingModelWrapper
from ..session.plt_wrapper.vision import VisionModelWrapper


class InputIterator:
    def __init__(
        self,
        model_name,
        task,
        dataset_name,
        quant_fn,
        quant_config_kwargs,
        workers=4,
        tokenizer=None,
    ) -> None:
        self._data_module = None
        self.train_loader_iter = None
        self.quantize_for_hw = partial(quant_fn, **quant_config_kwargs)
        self._init_data_module(
            dataset_name=dataset_name, workers=workers, tokenizer=tokenizer
        )
        self.reset()
        self._set_model_wrapper_cls(model_name=model_name, task=task)

    def _init_data_module(self, dataset_name, workers, tokenizer):
        data_module = MyDataModule(
            name=dataset_name,
            batch_size=1,
            workers=workers,
            tokenizer=tokenizer,
            max_token_len=512,
        )
        data_module.prepare_data()
        data_module.setup("fit")
        self._data_module = data_module

    def reset(self):
        self.train_loader_iter = iter(self._data_module.train_dataloader())

    def _set_model_wrapper_cls(self, model_name, task):
        self._model_wrapper_cls = get_model_wrapper(model_name, task)

    def __iter__(self):
        return self

    def __next__(self):
        if self._model_wrapper_cls == VisionModelWrapper:
            batch_x, _ = next(self.train_loader_iter)
            sw_input_args = [batch_x[[0], ...]]
            hw_input_args = [self.quantize_for_hw(batch_x[[0], ...])]
        elif self._model_wrapper_cls == NLPClassificationModelWrapper:
            batch = next(self.train_loader_iter)
            sw_input_args = [
                batch["input_ids"][[0], ...],
                batch["attention_mask"][[0], ...],
            ]
            # TODO: taking input_ids as hw input may not be practical. Maybe taking embedding output as input to hw?
            hw_input_args = [
                batch["input_ids"][[0], ...],
                batch["attention_mask"][[0], ...],
            ]

        elif self._model_wrapper_cls == NLPLanguageModelingModelWrapper:
            batch = next(self.train_loader_iter)
            sw_input_args = [
                batch["input_ids"][[0], ...],
                batch["attention_mask"][[0], ...],
                batch["labels"][[0], ...],
            ]
            # TODO: taking input_ids as hw input may not be practical. Maybe taking embedding output as input to hw?
            hw_input_args = [
                batch["input_ids"][[0], ...],
                batch["attention_mask"][[0], ...],
                batch["labels"][[0], ...],
            ]

        else:
            # NLPTranslation
            batch = next(self.train_loader_iter)
            sw_input_args = [
                batch["input_ids"][[0], ...],
                batch["attention_mask"][[0], ...],
                batch["decoder_input_ids"][[0], ...],
                batch["decoder_attention_mask"][[0], ...],
            ]
            # TODO: taking input_ids as hw input may not be practical. Maybe taking embedding output as input to hw?
            hw_input_args = [
                batch["input_ids"][[0], ...],
                batch["attention_mask"][[0], ...],
                batch["decoder_input_ids"][[0], ...],
                batch["decoder_attention_mask"][[0], ...],
            ]

        return sw_input_args, hw_input_args
