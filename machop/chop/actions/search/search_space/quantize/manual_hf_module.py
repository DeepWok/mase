# this is the search space for manual quantization of HuggingFace's PreTrainedModel
# manual quantization refers to the models named as <model_arch>_quantized at mase-tools/machop/chop/models/manual
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

from ..base import SearchSpaceBase
from ..utils import flatten_dict, unflatten_dict
from .....models.manual import get_manual_model_quant_config_parser

DEFAULT_QUANTIZATION_CONFIG = {
    "name": "integer",
    "bypass": True,
    "bias_frac_width": 5,
    "bias_width": 8,
    "data_in_frac_width": 5,
    "data_in_width": 8,
    "weight_frac_width": 3,
    "weight_width": 8,
}


class ManualHFModuleSearchSpaceMixedPrecisionPTQ(SearchSpaceBase):
    def _post_init_setup(self):
        assert isinstance(self.model, PreTrainedModel)

        # HuggingFace's PreTrainedModel
        self.model_cls = self.model.__class__
        # HuggingFace's PretrainedConfig
        self.model_config_cls = self.model.config.__class__
        # HuggingFace's PretrainedConfig._name_or_path (checkpoint name)
        self.model_name = self.model.config._name_or_path

        self.model_config: PretrainedConfig = self.model.config
        del self.model
        self.model = None  # we don't need the model anymore

        self.default_quant_config = DEFAULT_QUANTIZATION_CONFIG
        self.quant_config_parser = get_manual_model_quant_config_parser(
            config_cls=self.model_config_cls
        )

    def _rebuild_config(self, quant_config):
        config = self.model_config_cls.from_pretrained(
            self.model_name, quant_config=quant_config
        )
        return config

    @property
    def _num_hidden_layers(self):
        return self.model_config.num_hidden_layers

    def rebuild_model(self, quant_config: dict, is_eval_mode: bool):
        config = self._rebuild_config(quant_config)
        if "cuda" in self.accelerator.type:
            if self.config["setup"]["model_parallel"]:
                with init_empty_weights():
                    model = self.model_cls(config)
                device_map = infer_auto_device_map(
                    model,
                    no_split_module_classes=model._no_split_modules,
                )
                model = load_checkpoint_and_dispatch(
                    model, checkpoint=self.model_name, device_map=device_map
                )
            else:
                model = self.model_cls.from_pretrained(
                    self.model_name, config=config
                ).to(self.accelerator)
        elif self.accelerator.type == "cpu":
            model = self.model_cls.from_pretrained(self.model_name, config=config)
        else:
            raise ValueError(f"Unknown device: {self.accelerator}")

        if is_eval_mode:
            model.eval()
        return model

    def _strip_quant_name_entry(self, config):
        """
        strip the quantization name entry from the config or the quantization config parser cannot parse it
        ```text
        {                                              {
            "seq_0.conv1": {                                "seq_0.conv1": {
                "name": ["integer"],                            "name": "integer",
                "data_in_width": [5, 6, 7]    -->               "data_in_width": [5, 6, 7]
                ...                                             ...
            }                                                }
        }                                              }
        ```
        """
        stripped_config = {}
        flatten_dict(config, stripped_config)
        for k, v in stripped_config.items():
            if k.endswith("name"):
                assert isinstance(
                    v, (list, tuple)
                ), f"Quantization name must be a list: {v}, but got {type(v)}"
                assert (
                    len(v) == 1
                ), f"Search for quantization arithmetic is not supported. Only one quantization name is allowed: {v}"
                stripped_config[k] = v[0]
        return unflatten_dict(stripped_config)

    def _unstrip_quant_name_entry(self, config):
        unstripped_config = {}
        flatten_dict(config, unstripped_config)
        for k, v in unstripped_config.items():
            if k.endswith("name"):
                assert isinstance(v, str)
                unstripped_config[k] = [v]
        return unflatten_dict(unstripped_config)

    def build_search_space(self):
        seed = self.config["seed"]
        seed = self._strip_quant_name_entry(seed)
        parsed_choices = self.quant_config_parser(seed, self._num_hidden_layers)
        parsed_choices = self._unstrip_quant_name_entry(parsed_choices)
        flatten_dict(parsed_choices, flattened=self.choices_flattened)
        self.choice_lengths_flattened = {
            k: len(v) for k, v in self.choices_flattened.items()
        }

    def flattened_indexes_to_config(
        self, indexes: dict[str, int]
    ) -> dict[str, dict[str, int]]:
        flattened_config = {}
        for k, v in indexes.items():
            flattened_config[k] = self.choices_flattened[k][v]
        config = unflatten_dict(flattened_config)
        config["default"] = self.default_quant_config
        return config
