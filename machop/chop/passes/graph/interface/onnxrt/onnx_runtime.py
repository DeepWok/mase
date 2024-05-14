import logging
import torch
import os
from datetime import datetime
from pathlib import Path
from prettytable import PrettyTable
import onnx
import onnxruntime as ort
from ...transforms.onnxrt.quantize import Quantizer


def onnx_runtime_interface_pass(graph, pass_args=None):
    """
    Converts a PyTorch model within a MaseGraph to ONNX format and performs quantization as specified in the configuration.

    This function facilitates the conversion of a PyTorch model to ONNX format, leveraging ONNX Runtime (ONNXRT) for potential quantization and optimization. Depending on the `precision` parameter set in `passes.onnxruntime.default.config`, the model can be quantized to various numeric precisions including INT8, UINT8, INT16, UINT16, or FP16, affecting the model's performance and latency. Notably, INT8 and UINT8 quantization typically yield significant latency improvements at the potential cost of reduced model performance.

    :param graph: The model graph to be processed and quantized.
    :type graph: MaseGraph
    :param pass_args: Configuration arguments for the ONNX Runtime session, specifying details such as quantization type, precision, and calibration data.
    :type pass_args: dict, optional
    :return: A tuple containing the original graph and a dictionary with the path to the quantized ONNX model and quantization metadata.
    :rtype: tuple(MaseGraph, dict)

    The quantization process supports three methodologies:
    - **Static Quantization**: Utilizes a calibration dataset to calculate scale and zero points for activations offline, providing consistent quantization parameters across all forward passes. Requires setting `num_calibration_batches` to define the calibration dataset size.
    - **Dynamic Quantization**: Calculates scale and zero points on-the-fly during each forward pass, offering potentially higher accuracy at the cost of additional computational overhead.
    - **Auto Mixed Precision Quantization**: Automatically balances FP16 and FP32 precisions to maintain accuracy levels, particularly suitable for GPU execution. This process is automatic and requires no specific precision setting in the configuration.

    Please note that not all modules may be supported for the specified quantization types. Refer to the [ONNX Runtime documentation](https://onnxruntime.ai/docs/) for more information.

    The function also prepares the model by applying pre-processing optimizations before quantization, storing intermediate models in a designated `pre_processed` directory. The final quantized models are stored in a structured directory format under `mase_output/onnxrt`.

    Directory structure for model storage:
    ```
    mase_output
    └── onnxrt
        └── model_task_dataset_date
            ├── optimized
            ├── pre_processed
            ├── static_quantized
            └── dynamic_quantized
    ```

    Example of usage:

        graph = MaseGraph(...)
        processed_graph, metadata = onnx_runtime_interface_pass(graph, {'precision': 'int8', 'quantization_type': 'static'})

    This example demonstrates how to invoke the ONNX Runtime interface pass, specifying INT8 precision and static quantization type.
    """
    onnx_runtime_session = ONNXRuntime(config=pass_args)
    pytorch_model = graph.model

    onnx_model_path = onnx_runtime_session.pytorch_to_onnx(pytorch_model)
    onnx_model_graph = onnx_runtime_session.load_onnx(onnx_model_path).graph
    onnx_runtime_session.summarize_ONNX_graph(onnx_model_graph)
    quant_meta = onnx_runtime_session.quantize(onnx_model_path)

    return graph, {"onnx_path": onnx_model_path, **quant_meta}


class ONNXRuntime:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def _prepare_save_path(self, quantized_type: str):
        """Creates and returns a save path for the model."""
        root = Path(__file__).resolve().parents[6]
        current_date = datetime.now().strftime("%Y-%m-%d")
        model_dir = f'{self.config["model"]}_{self.config["task"]}_{self.config["dataset"]}_{current_date}'
        save_dir = root / f"mase_output/onnxrt/{model_dir}/{quantized_type}"
        save_dir.mkdir(parents=True, exist_ok=True)

        existing_versions = len(os.listdir(save_dir))
        version = (
            "version_0" if existing_versions == 0 else f"version_{existing_versions}"
        )

        save_dir = save_dir / version
        save_dir.mkdir(parents=True, exist_ok=True)

        return save_dir / f"model.onnx"

    def pytorch_to_onnx(self, model):
        """Converts PyTorch model to ONNX format and saves it."""
        self.logger.info("Converting PyTorch model to ONNX...")
        save_path = self._prepare_save_path("optimized")
        self.logger.info(f"Project will be created at {save_path.parent.parent.parent}")

        # ensure model is on the appropriate device
        model = model.to(self.config["accelerator"])

        dataloader = self.config["data_module"].train_dataloader()
        train_sample = next(iter(dataloader))[0]
        train_sample = train_sample.to(self.config["accelerator"])

        torch.onnx.export(
            model,
            train_sample,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
        )
        self.logger.info(f"ONNX Conversion Complete. Stored ONNX model to {save_path}")

        self.onnx_path = save_path

        return save_path

    def summarize_ONNX_graph(self, graph):
        # Initialize a PrettyTable to display the summary
        summary_table = PrettyTable()
        summary_table.field_names = [
            "Index",
            "Name",
            "Type",
            "Inputs",
            "Outputs",
            "Attributes",
        ]

        # Parse through the model's graph
        for index, node in enumerate(graph.node):
            # Gather node information
            node_name = node.name or f"Node_{index}"  # Some nodes might not have names
            node_type = node.op_type
            inputs = [str(input) for input in node.input]
            outputs = [str(output) for output in node.output]
            attributes = [attr.name for attr in node.attribute]

            # Add information to the table
            summary_table.add_row(
                [
                    index,
                    node_name,
                    node_type,
                    ", ".join(inputs),
                    ", ".join(outputs),
                    ", ".join(attributes),
                ]
            )
        self.logger.info(f"ONNX Model Summary: \n{summary_table}")

    def quantize(self, model_path) -> dict:
        # only quantize is set in the default config
        try:
            self.config["default"]["config"]["quantize"]
        except:
            self.logger.warning(
                "Quantization is not set in default config. Skipping quantization."
            )
            return {}

        if not self.config["default"]["config"]["quantize"]:
            return {}

        quantizer = Quantizer(self.config)
        try:
            quant_types = self.config["default"]["config"]["quantize_types"]
        except (TypeError, KeyError):
            quant_types = ["static"]

        # Pre-process the model adding further optimizations and store to prep_path
        prep_path = self._prepare_save_path("pre_processed")
        quantizer.pre_process(model_path, prep_path)
        quant_models = {}
        for quant_type in quant_types:
            match quant_type:
                case "static":
                    quantized_path = self._prepare_save_path("static_quantized")
                    quantizer.quantize_static(prep_path, quantized_path)
                    quant_models["onnx_static_quantized_path"] = quantized_path
                case "dynamic":
                    quantized_path = self._prepare_save_path("dynamic_quantized")
                    quantizer.quantize_dynamic(prep_path, quantized_path)
                    quant_models["onnx_dynamic_quantized_path"] = quantized_path
                case "auto":
                    quantized_path = self._prepare_save_path("auto_quantized")
                    quantizer.quantize_auto_mixed_precision(prep_path, quantized_path)
                    quant_models["onnx_auto_mixed_precision_path"] = quantized_path
                case _:
                    raise Exception(f"Invalid quantization type: {quant_type}")
        return quant_models

    def load_onnx(self, onnx_model_path):
        """Load .onnx model"""

        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)

        return onnx_model
