from importlib.util import find_spec
from copy import copy, deepcopy
import logging
import torch
import onnx
import numpy as np

logger = logging.getLogger(__name__)

pytorch_quantization_is_installed = False

if find_spec("pytorch_quantization") is None or find_spec("tensorrt") is None:

    def tensorrt_engine_interface_pass(graph, pass_args=None):
        raise ImportError(
            "tensorrt or pytorch_quantization is not installed. Cannot use tensorrt quantize pass."
        )

    def Quantizer(config):
        raise ImportError(
            "pytorch_quantization is not installed. Cannot use tensorrt quantize pass."
        )

else:
    import tensorrt as trt
    from pytorch_quantization import quant_modules, calib
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization.nn import TensorQuantizer
    from pytorch_quantization.tensor_quant import QuantDescriptor

    from chop.passes.graph.utils import (
        get_mase_op,
        get_mase_type,
        get_node_actual_target,
    )
    from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass
    from ...utils import deepcopy_mase_graph
    from ...transforms.tensorrt.quantize.utils import (
        Int8Calibrator,
        prepare_save_path,
        check_for_value_in_dict,
    )

    def tensorrt_engine_interface_pass(graph, pass_args=None):
        """
        Converts a PyTorch model within a MaseGraph to a TensorRT engine, optimizing the model for faster inference speeds.

        This function acts as an interface between PyTorch and TensorRT, leveraging ONNX as an intermediate format. It's designed to be used post-Quantization Aware Training (QAT), ensuring that the quantized model can benefit from the performance enhancements offered by TensorRT. The conversion process saves the resulting ONNX and TensorRT engine files in their respective directories, facilitating easy deployment and version control.

        :param graph: The model graph to be converted. This graph should represent a model that has already been quantized and fine-tuned.
        :type graph: MaseGraph
        :param pass_args: A dictionary containing arguments that may affect the conversion process, such as optimization levels or specific TensorRT flags.
        :type pass_args: dict, optional
        :return: A tuple containing the graph with its model now linked to the generated TensorRT engine, and a dictionary with paths to the ONNX and TensorRT engine files.
        :rtype: tuple(MaseGraph, dict)

        The conversion process involves two main steps:
        1. Exporting the PyTorch model to ONNX format.
        2. Converting the ONNX model to a TensorRT engine.

        The paths to the saved `.onnx` and `.trt` files are included in the return value to provide easy access for subsequent deployment or analysis.
        The resulting files are are saved in the following directory structure, facilitating easy access and version control:

        - mase_output
            - tensorrt
                - quantization
                    - model_task_dataset_date
                        - cache
                        - ckpts
                            - fine_tuning
                        - json
                        - onnx
                        - trt

        Example of usage:

            graph = MaseGraph(...)
            converted_graph, paths = tensorrt_engine_interface_pass(graph, pass_args={})

        This example initiates the conversion of a quantized and fine-tuned PyTorch model to a TensorRT engine, with the paths to the resulting ONNX and TRT files being returned for further use.

        Note:
        The `.onnx` and `.trt` files are stored according to the directory structure outlined in Section 1.3 of the Quantized Aware Training (QAT) documentation, ensuring organized and accessible storage for these critical files.
        """
        quantizer = Quantizer(pass_args)
        trt_engine_path, onnx_path = quantizer.pytorch_to_trt(graph)

        # Link the model with the graph for further operations or evaluations
        graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)

        return graph, {"trt_engine_path": trt_engine_path, "onnx_path": onnx_path}

    class Quantizer:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(__name__)
            logging.getLogger("pytorch_quantization").setLevel(logging.ERROR)

        def pytorch_to_trt(self, graph):
            """Converts PyTorch model to TensorRT format."""
            # Model is first converted to ONNX format and then to TensorRT
            ONNX_path = self.pytorch_to_ONNX(graph.model)
            TRT_path = self.ONNX_to_TRT(ONNX_path)
            self.export_TRT_model_summary(TRT_path)

            return TRT_path, ONNX_path

        def get_config(self, name: str):
            """Retrieve specific configuration from the instance's config dictionary or return default."""
            return self.config.get(name, "default")

        def pre_quantization_test(self, model):
            """Evaluate pre-quantization performance."""
            print("Evaluate pre-quantization performance...")
            # Add evaluation code here

        def pytorch_quantize(self, graph):
            """Applies quantization procedures to PyTorch graph based on type."""
            # Add quantization code here

        def ONNX_to_TRT(self, ONNX_path):
            self.logger.info("Converting PyTorch model to TensorRT...")

            # Check for layer wise mixed precision
            layer_wise_mixed_precision = (
                True
                if check_for_value_in_dict(self.config, "int8")
                and check_for_value_in_dict(self.config, "fp16")
                else False
            )

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            with open(ONNX_path, "rb") as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        self.logger.error(parser.get_error(error))
                    raise Exception("Failed to parse the ONNX file.")

            # Create the config object here
            config = builder.create_builder_config()
            config.max_workspace_size = 4 << 30  # 4GB

            default_precision = self.config["default"]["config"]["precision"]

            # This section may be uncommented if pytorch-quantization is not used for int8 Calibration
            """
            # Only required if pytorch-quantization is not used
            config.set_flag(trt.BuilderFlag.INT8)
            if default_precision == 'int8':
                config.int8_calibrator = Int8Calibrator(
                    self.config['num_calibration_batches'],
                    self.config['data_module'].train_dataloader(),
                    prepare_save_path(self.config, method='cache', suffix='cache')
                    )
            """

            # Only quantize and calibrate non int8 pytorch-quantization
            if default_precision != "int8":
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
                config.set_flag(trt.BuilderFlag.DIRECT_IO)
                config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            if default_precision == "fp16" and not layer_wise_mixed_precision:
                config.set_flag(trt.BuilderFlag.FP16)

            elif layer_wise_mixed_precision:
                # Now, iterate over the network layers and set precision as per the config
                for idx in range(network.num_layers):
                    layer = network.get_layer(idx)
                    layer_key = f"feature_layers_{idx}"
                    layer_precision = (
                        self.config.get("passes", {})
                        .get("tensorrt", {})
                        .get(layer_key, {})
                        .get("config", {})
                        .get("precision", default_precision)
                    )

                    # Apply precision settings based on the layer_precision value
                    if layer_precision == "fp16":
                        layer.precision = trt.float16
                        layer.set_output_type(0, trt.DataType.HALF)
                    elif layer_precision == "int8":
                        layer.precision = trt.int8
                        layer.set_output_type(0, trt.DataType.INT8)
                    else:
                        # You might want to handle the default case or unsupported precision types differently
                        print(
                            f"Warning: Unsupported precision type '{layer_precision}' for layer {idx}. Defaulting to fp16."
                        )
                        layer.precision = trt.float16
                        layer.set_output_type(0, trt.DataType.HALF)

            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise Exception(
                    "Failed to build serialized network. A builderflag or config parameter may be incorrect or the ONNX model is unsupported."
                )

            trt_path = prepare_save_path(self.config, method="trt", suffix="trt")
            with open(trt_path, "wb") as f:
                f.write(serialized_engine)

            # Optimization profiles are needed for dynamic input shapes.
            profile = builder.create_optimization_profile()
            inputTensor = network.get_input(0)
            profile.set_shape(
                inputTensor.name,
                (1,) + inputTensor.shape[1:],
                (8,) + inputTensor.shape[1:],
                (32,) + inputTensor.shape[1:],
            )
            config.add_optimization_profile(profile)

            self.logger.info(
                f"TensorRT Conversion Complete. Stored trt model to {trt_path}"
            )
            return trt_path

        def pytorch_to_ONNX(self, model):
            """Converts PyTorch model to ONNX format and saves it."""
            self.logger.info("Converting PyTorch model to ONNX...")

            onnx_path = prepare_save_path(self.config, method="onnx", suffix="onnx")

            dataloader = self.config["data_module"].train_dataloader()
            train_sample = next(iter(dataloader))[0]
            train_sample = train_sample.to(self.config["accelerator"])

            """
            This line may produce the warning if the model input size is fixed:
                    torch.onnx.export(model.cuda(), train_sample.cuda(), onnx_path, export_params=True, opset_version=11,
                            do_constant_folding=True, input_names=['input'])# Load the ONNX model
            It is a known issue: https://github.com/onnx/onnx/issues/2836 https://github.com/ultralytics/yolov5/issues/5505
            """
            torch.onnx.export(
                model.cuda(),
                train_sample.cuda(),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
            )  # Load the ONNX model

            model = onnx.load(onnx_path)
            try:
                onnx.checker.check_model(model)
            except onnx.checker.ValidationError as e:
                raise Exception(f"ONNX Conversion Failed: {e}")

            self.logger.info(
                f"ONNX Conversion Complete. Stored ONNX model to {onnx_path}"
            )
            return onnx_path

        def export_TRT_model_summary(self, TRT_path):
            """Saves TensorRT model summary to json"""
            with open(TRT_path, "rb") as f:
                trt_engine = trt.Runtime(
                    trt.Logger(trt.Logger.ERROR)
                ).deserialize_cuda_engine(f.read())
                inspector = trt_engine.create_engine_inspector()

                # Retrieve engine information in JSON format
                layer_info_json = inspector.get_engine_information(
                    trt.LayerInformationFormat.JSON
                )

                # Save the engine information to a JSON file
                json_filename = prepare_save_path(
                    self.config, method="json", suffix="json"
                )
                with open(json_filename, "w") as json_file:
                    json_file.write(layer_info_json)
            self.logger.info(f"TensorRT Model Summary Exported to {json_filename}")
