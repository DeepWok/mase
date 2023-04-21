from collections import OrderedDict
from functools import partial

import torch

from ...dataset import MyDataModule
from ...modify.quantizers.quantizers_for_hw import integer_quantizer_for_hw
from ...session.plt_wrapper import get_model_wrapper
from ...session.plt_wrapper.nlp.classification import NLPClassificationModelWrapper
from ...session.plt_wrapper.nlp.lm import NLPLanguageModelingModelWrapper
from ...session.plt_wrapper.vision import VisionModelWrapper

QUANTIZER_FOR_HW_MAPPING = {"fixed": integer_quantizer_for_hw}


class InputGenerator:
    def __init__(
        self,
        model_name: str,
        task: str,
        dataset_name: str,
        quant_name: str,
        quant_config_kwargs: dict,
        trans_num: int = 8,
        workers: int = 4,
        tokenizer=None,
    ) -> None:
        """
        quant_name (str): the string name of quantization method for quantizing input data.
                          'integer' for fixed-point quantization
        # TODO precision format needs to synchronised
        quant_config_kwargs (Dict): the config of quantization method for input data.
                          For example, {"width": 8, "frac_width": 6} for fixed-point quantized input data.
        """
        self._train_dataset = None
        self.quantize_for_hw = partial(
            QUANTIZER_FOR_HW_MAPPING[quant_name], **quant_config_kwargs
        )
        self.trans_num = trans_num
        self._init_train_dataset(
            dataset_name=dataset_name,
            trans_num=trans_num,
            workers=workers,
            tokenizer=tokenizer,
        )
        # self.reset()
        self._set_model_wrapper_cls(model_name=model_name, task=task)

    def _init_train_dataset(self, dataset_name, trans_num, workers, tokenizer):
        data_module = MyDataModule(
            dataset_name=dataset_name,
            batch_size=trans_num,
            workers=workers,
            tokenizer=tokenizer,
            max_token_len=512,
        )
        data_module.prepare_data()
        data_module.setup("fit")
        self._train_dataset = data_module.train_dataset

    def _set_model_wrapper_cls(self, model_name, task):
        self._model_wrapper_cls = get_model_wrapper(model_name, task)

    def _tensor_batch_to_list(self, x: torch.Tensor):
        list_of_samples = []
        for samples_index in range(self.trans_num):
            list_of_samples.append(x[samples_index, ...])
        return list_of_samples

    def __getitem__(self, trans_index):
        start_index = trans_index * self.trans_num
        end_index = (trans_index + 1) * self.trans_num

        for data_index in range(start_index, end_index):
            data_i = self._train_dataset[data_index]
            if self._model_wrapper_cls == VisionModelWrapper:
                data_i_x, _ = data_i
                if data_index == start_index:
                    batch_x = data_i_x.unsqueeze(0)
                else:
                    batch_x = torch.cat([batch_x, data_i_x.unsqueeze(0)], dim=0)
            else:
                raise NotImplementedError

        if self._model_wrapper_cls == VisionModelWrapper:
            input_args_sw = [batch_x]
            batch_x_hw = self._tensor_batch_to_list(self.quantize_for_hw(batch_x))
            input_args_hw = [batch_x_hw]
        else:
            raise NotImplementedError

        return input_args_sw, input_args_hw

    def __call__(self):
        """
        A temporary solution for Toy

        returns a list of tensor and a list of tuples
        """
        input_args_sw, input_args_hw = self.__getitem__(0)
        batch_x = input_args_sw[0]
        input_args_sw = []
        for i in range(self.trans_num):
            input_args_sw.append(batch_x[[i], ...])
        return input_args_sw, input_args_hw[0]

    # def __next__(self):
    #     if self._model_wrapper_cls == VisionModelWrapper:
    #         batch_x, _ = next(self.train_loader)
    #         sw_input_args = [batch_x[[0], ...]]
    #         hw_input_args = [self.quantize_for_hw(batch_x[[0], ...])]
    #     elif self._model_wrapper_cls == NLPClassificationModelWrapper:
    #         batch = next(self.train_loader)
    #         sw_input_args = [
    #             batch["input_ids"][[0], ...],
    #             batch["attention_mask"][[0], ...],
    #         ]
    #         # TODO: taking input_ids as hw input may not be practical. Maybe taking embedding output as input to hw?
    #         hw_input_args = [
    #             batch["input_ids"][[0], ...],
    #             batch["attention_mask"][[0], ...],
    #         ]

    #     elif self._model_wrapper_cls == NLPLanguageModelingModelWrapper:
    #         batch = next(self.train_loader)
    #         sw_input_args = [
    #             batch["input_ids"][[0], ...],
    #             batch["attention_mask"][[0], ...],
    #             batch["labels"][[0], ...],
    #         ]
    #         # TODO: taking input_ids as hw input may not be practical. Maybe taking embedding output as input to hw?
    #         hw_input_args = [
    #             batch["input_ids"][[0], ...],
    #             batch["attention_mask"][[0], ...],
    #             batch["labels"][[0], ...],
    #         ]

    #     else:
    #         # NLPTranslation
    #         batch = next(self.train_loader)
    #         sw_input_args = [
    #             batch["input_ids"][[0], ...],
    #             batch["attention_mask"][[0], ...],
    #             batch["decoder_input_ids"][[0], ...],
    #             batch["decoder_attention_mask"][[0], ...],
    #         ]
    #         # TODO: taking input_ids as hw input may not be practical. Maybe taking embedding output as input to hw?
    #         hw_input_args = [
    #             batch["input_ids"][[0], ...],
    #             batch["attention_mask"][[0], ...],
    #             batch["decoder_input_ids"][[0], ...],
    #             batch["decoder_attention_mask"][[0], ...],
    #         ]

    #     return sw_input_args, hw_input_args
