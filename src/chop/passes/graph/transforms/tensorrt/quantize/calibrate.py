import logging

import torch
from torch.autograd import Variable


import pytorch_quantization.calib as calib
import pytorch_quantization.nn as qnn
import tensorrt as trt
from cuda import cudart
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor

from .utils import FakeQuantizer, check_for_value_in_dict, get_calibrator_dataloader


def tensorrt_fake_quantize_transform_pass(graph, pass_args=None):
    """
    Applies a fake quantization pass to a model graph, preparing it for calibration and fine-tuning before actual quantization.

    This pass simulates quantization effects on the model's precision by modifying its layers to fake quantized versions, based on the `pass_args`. It's a preliminary step for models specifically targeting int8 calibration, as other precisions are not supported by the `pytorch-quantization` toolkit. This process is crucial for achieving accurate model quantization without significant loss in precision.

    :param graph: The model graph to be transformed.
    :type graph: MaseGraph
    :param pass_args: A dictionary containing arguments that define how the transformation is applied. The key "by" determines whether the fake quantization should be applied by layer "type" or "name". Additional parameters required for the FakeQuantizer can also be passed.
    :type pass_args: dict, optional
    :return: A tuple containing the transformed graph and an empty dictionary. The empty dictionary is a placeholder for potential future use.
    :rtype: tuple(MaseGraph, dict)

    The fake quantization can target specific layers, including Linear, Conv1d/2d/3d, ConvTranspose1d/2d/3d, MaxPool1d/2d/3d, AvgPool1d/2d/3d, LSTM, and LSTMCell. This ensures that the most impactable layers for quantization are addressed, preparing the model for int8 calibration effectively.

    Example of usage:

        graph = MaseGraph(...)
        transformed_graph, _ = tensorrt_fake_quantize_transform_pass(graph, {'by': 'type'})

    This example demonstrates initiating a fake quantization transformation by layer type. Layers are replaced with their fake quantized counterparts if they are recognized as quantizable based on the specified criteria in `pass_args`.

    For more information on creating custom quantized modules or understanding the `pytorch-quantization` toolkit, refer to NVIDIA's documentation: https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html

    Raises:
        ValueError: If the "by" parameter in `pass_args` is not supported. Currently, only "type" and "name" are valid options for specifying how to apply the fake quantization.
    """
    by = pass_args["by"]
    fq = FakeQuantizer(pass_args)
    match by:
        case "type":
            graph = fq.fake_quantize_by_type(graph)
        case "name":
            graph = fq.fake_quantize_by_name(graph)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    return graph, {}


def tensorrt_calibrate_transform_pass(graph, pass_args=None):
    """
    Performs calibration on a model graph by deciding the best maximum absolute values (amax) for activations using specified calibrators.

    Calibration is a critical step in the quantization process, ensuring that the quantized model maintains accuracy close to the original model by optimizing the scale factors for activations. This function utilizes a `Calibrator` object, which takes `pass_args` to determine the calibration method and parameters, and applies it to the entire graph.

    :param graph: The model graph to be calibrated.
    :type graph: MaseGraph
    :param pass_args: A dictionary containing arguments for calibration, including the choice of calibrator and any specific parameters it requires. Supported calibrators are specified in a TOML configuration and can include methods like "percentile", "mse", and "entropy".
    :type pass_args: dict, optional
    :return: A tuple containing the calibrated graph and an empty dictionary. The empty dictionary is a placeholder for potential extensions.
    :rtype: tuple(MaseGraph, dict)

    Note on calibrators:
    - "percentile" calibration requires a list of percentiles to determine the best amax values.
    - "max" calibration simplifies the process by using the global maximum absolute value across the entire dataset for calibration. This requires removing "histogram" weight and input calibrators from the configuration and replacing them with "max".
    - Optionally, if "post_calibration_analysis" is enabled in the `pass_args`, a subsequent analysis pass (`tensorrt_analysis_pass`) can be automatically triggered to evaluate the effectiveness of each calibrator.

    Example of usage:

        graph = MaseGraph(...)
        calibrated_graph, _ = tensorrt_calibrate_transform_pass(graph, {'calibrators': ["percentile", "mse", "entropy"], 'percentiles': [99.9], 'post_calibration_analysis': True})

    This example shows how to calibrate a model graph using a range of calibrators and additional parameters such as percentiles for the "percentile" method and enabling post-calibration analysis.

    It's important to choose the right calibrator based on the model and dataset characteristics to ensure optimal performance and accuracy of the quantized model.
    """
    calibrator = Calibrator(pass_args)
    graph = calibrator.calibrate_model(graph)
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}


class Calibrator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def get_config(self, config: dict, name: str):
        """Retrieve specific configuration from the config dictionary or return default."""
        return config.get(name, config["default"])["config"]

    def eval_calibration(self, graph, calibrator):
        """Performs post calibration analysis for the given calibrator."""
        from chop.passes.graph import runtime_analysis_pass

        self.logger.info(
            f"Performing post calibration analysis for calibrator {calibrator}..."
        )

        # Temporarily change 'test' in config for evaluation on the validation set, not the test set
        config_test = self.config.get("test", False)  # Store original 'test' value
        self.config["test"] = False  # Ensure evaluation is not on test set

        runtime_analysis_pass(graph, pass_args=self.config)  # Run the analysis pass

        # Restore original 'test' configuration after evaluation
        self.config["test"] = config_test

        self.logger.info("Post calibration analysis complete.")

    def compute_amax(self, model, **kwargs):
        """Computes and loads the maximum activation values for quantization calibration."""
        # Load calibration result
        for name, module in model.named_modules():
            if isinstance(module, qnn.TensorQuantizer):
                if module._calibrator is not None:
                    # Load calibration max values depending on the calibrator type
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax()
                    else:
                        module.load_calib_amax(**kwargs)
                self.logger.info(f"{name:40}: {module}")
        model.cuda()

    def calibrate_model(self, graph):
        """Performs the calibration pass on the model using the given data loader."""

        if not check_for_value_in_dict(self.config, "int8"):
            self.logger.warning(
                "int8 precision not found in config. Skipping calibration."
            )
            return graph

        self.logger.info("Starting calibration of the model in PyTorch...")
        quant_modules.initialize()
        graph.model.cuda()

        with torch.no_grad():
            # Turn on calibration tool
            for name, module in graph.model.named_modules():
                if isinstance(module, qnn.TensorQuantizer):
                    if module._calibrator is not None:
                        self.logger.info(
                            "Disabling Quantization and Enabling Calibration"
                        )
                        module.disable_quant()
                        module.enable_calib()
                    else:
                        module.disable()

            # Create a calibrator_dataloader that is a subset of the training dataloader
            # Number of batches defined in the config by num_calibration_batches
            calibrator_dataloader = get_calibrator_dataloader(
                self.config["data_module"].train_dataloader(),
                self.config.get("num_calibration_batches", 200),
            )

            for i, (xTrain, _) in enumerate(calibrator_dataloader):
                graph.model(Variable(xTrain).cuda())

            # Turn off calibration tool
            for _, module in graph.model.named_modules():
                if isinstance(module, qnn.TensorQuantizer):
                    if module._calibrator is not None:
                        module.enable_quant()
                        self.logger.info(
                            "Enabling Quantization and Disabling Calibration"
                        )
                        module.disable_calib()
                    else:
                        module.enable()

            # Apply the specific calibration based on user input
            try:
                calibs = self.config.get("default")["config"]["calibrators"]
            except KeyError:
                calibs = "entropy"
            for calib in calibs:
                match calib:
                    case "entropy":
                        self.compute_amax(graph.model, method=calib)
                    case "percentile":
                        try:
                            percentiles = self.config.get("default")["config"][
                                "percentiles"
                            ]
                        except KeyError:
                            percentiles = [99]
                        for percentile in percentiles:
                            self.compute_amax(
                                graph.model, method=calib, percentile=percentile
                            )
                            # perform an analysis pass if required
                            if self.config["post_calibration_analysis"]:
                                self.eval_calibration(graph, f"{calib}_{percentile}")
                        continue
                    case "mse":
                        self.compute_amax(graph.model, method=calib)

                # perform an analysis pass if required
                if self.config["post_calibration_analysis"]:
                    self.eval_calibration(graph, calib)

            self.logger.info("Succeeded in calibrating the model in PyTorch!")
            return graph
