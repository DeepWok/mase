import logging
import torch
from pathlib import PosixPath
from chop.ir import MaseGraph
from .utils import PowerMonitor, get_execution_provider
import os
from tabulate import tabulate
import torchmetrics
import numpy as np
import onnxruntime as ort
import json
from datetime import datetime
from pathlib import Path
import time

import tensorrt as trt
from cuda import cudart


def runtime_analysis_pass(model, pass_args=None):
    """
    Evaluates the performance of a model by analyzing its inference speed, accuracy, and other relevant metrics.

    This function is part of the model optimization and evaluation pipeline, designed to showcase the improvements in inference speed achieved through quantization and conversion to TensorRT engines. It accepts models in various formats, including a `MaseGraph` object, a path to an ONNX model, or a path to a TensorRT engine, facilitating a flexible analysis process. The function outputs a comprehensive set of performance metrics that help in assessing the model's efficiency and effectiveness post-optimization.

    :param model: The model to be analyzed. Can be a `MaseGraph`, a `PosixPath` to an ONNX model, or a `PosixPath` to a TensorRT engine.
    :type model: MaseGraph or PosixPath
    :param pass_args: Optional arguments that may influence the analysis, such as specific metrics to be evaluated or configurations for the analysis tools.
    :type pass_args: dict, optional
    :return: A tuple containing the original model and a dictionary with the results of the analysis, including metrics such as accuracy, precision, recall, F1 score, loss, latency, GPU power usage, and inference energy consumption.
    :rtype: tuple(MaseGraph or PosixPath, dict)

    The analysis is conducted by creating an instance of `RuntimeAnalysis` with the model and optional arguments, evaluating the model's performance, and then storing the results. The metrics provided offer a holistic view of the model's operational characteristics, enabling a thorough comparison between the original unquantized model, the INT8 quantized model, and other variations that have undergone optimization processes.

    Example of usage:

        model_path = PosixPath('/path/to/model.trt')
        _, results = runtime_analysis_pass(model_path, pass_args={})

    This example demonstrates initiating the runtime analysis pass on a model provided via a TensorRT engine path. The function returns a set of metrics illustrating the model's performance characteristics, such as inference speed and accuracy.

    The performance metrics include:
    - Average Test Accuracy
    - Average Precision
    - Average Recall
    - Average F1 Score
    - Average Loss
    - Average Latency
    - Average GPU Power Usage
    - Inference Energy Consumption

    These metrics provide valuable insights into the model's efficiency, effectiveness, and operational cost, crucial for informed decision-making regarding model deployment in production environments.
    """
    analysis = RuntimeAnalysis(model, pass_args)
    results = analysis.evaluate()
    analysis.store(results)

    return model, results


class RuntimeAnalysis:
    def __init__(self, model, config):
        # Istantiate default performance analyzer args
        if "num_batches" not in config.keys():
            config["num_batches"] = 500
            config["num_GPU_warmup_batches"] = 5
            config["test"] = True

        self.config = config

        self.logger = logging.getLogger(__name__)
        self.num_of_classes = self.config["data_module"].dataset_info.num_classes

        match model:
            case MaseGraph():
                # Check if model is mase graph
                self.model = model.model
                self.model_name = self.config["model"]
                self.model_type = "mase_graph"

            case PosixPath() as path:
                match path.suffix:
                    case ".trt":
                        # Load the serialized TensorRT engine
                        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                        with open(path, "rb") as f:
                            self.engine = trt.Runtime(
                                TRT_LOGGER
                            ).deserialize_cuda_engine(f.read())
                        self.runtime = trt.Runtime(TRT_LOGGER)
                        self.num_io = self.engine.num_io_tensors
                        self.lTensorName = [
                            self.engine.get_tensor_name(i) for i in range(self.num_io)
                        ]
                        self.n_Input = [
                            self.engine.get_tensor_mode(self.lTensorName[i])
                            for i in range(self.num_io)
                        ].count(trt.TensorIOMode.INPUT)
                        self.context = self.engine.create_execution_context()
                        self.model = self.context
                        self.model_name = f"{self.config['model']}-trt_quantized"
                        self.model_type = "tensorrt"

                    case ".onnx":
                        # Load the exported ONNX model into an ONNXRuntime inference session
                        execution_provider = get_execution_provider(
                            self.config["accelerator"]
                        )
                        self.logger.info(
                            f"Using {execution_provider} as ONNX execution provider."
                        )

                        # Create a session options object
                        sess_options = ort.SessionOptions()

                        # Set the log severity level
                        # Levels are: VERBOSE = 0, INFO = 1, WARNING = 2, ERROR = 3, FATAL = 4
                        # Setting it to 1 will capture both INFO and more severe messages
                        sess_options.log_severity_level = 2

                        self.model = ort.InferenceSession(
                            path,
                            providers=execution_provider,
                            sess_options=sess_options,
                        )
                        self.model_name = f"{self.config['model']}-onnx"
                        self.model_type = "onnx"
                    case _:
                        # If file type is neither .trt nor .onnx
                        raise Exception(
                            "Model must be a MaseGraph or a path to a trt file. Have you run the quantization pass?"
                        )
            case _:
                # If model is neither MaseGraph nor PosixPath
                raise Exception(
                    "Model must be a MaseGraph or a PosixPath to a trt file. Have you run the quantization pass?"
                )

    def store(self, results):
        # Save the results in a JSON file
        save_path = self._prepare_save_path(self.model_type, "json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Runtime analysis results saved to {save_path}")

    def _prepare_save_path(self, method: str, suffix: str):
        """Creates and returns a save path for the model."""
        root = Path(__file__).resolve().parents[7]
        current_date = datetime.now().strftime("%Y-%m-%d")
        model_dir = f'{self.config["model"]}_{self.config["task"]}_{self.config["dataset"]}_{current_date}'
        save_dir = root / f"mase_output/tensorrt/quantization/{model_dir}/{method}"
        save_dir.mkdir(parents=True, exist_ok=True)

        existing_versions = len(os.listdir(save_dir))
        version = (
            "version_0" if existing_versions == 0 else f"version_{existing_versions}"
        )

        save_dir = save_dir / version
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"model.{suffix}"

    def _summarize(self):
        io_info_lines = [
            "Index | Type    | DataType | Static Shape         | Dynamic Shape        | Name",
            "------|---------|----------|----------------------|----------------------|-----------------------",
        ]

        for name in self.lTensorName:
            # Get the binding index for the tensor name
            binding_index = self.engine.get_binding_index(name)

            # Determine whether this is an input or an output
            io_type = "Input" if binding_index < self.n_Input else "Output"

            # Get data type and shapes using the binding index
            dtype = self.engine.get_binding_dtype(binding_index)
            static_shape = self.engine.get_binding_shape(binding_index)
            dynamic_shape = self.context.get_binding_shape(binding_index)

            # Translate TensorRT data type to string
            dtype_str = str(dtype).split(".")[
                -1
            ]  # Convert from DataType.FLOAT to FLOAT

            # Format the IO information
            io_info = f"{binding_index:<5} | {io_type:<7} | {dtype_str:<8} | {str(static_shape):<22} | {str(dynamic_shape):<22} | {name}"
            io_info_lines.append(io_info)

        # Join all IO information into a single string and log it
        io_info_str = "\n".join(io_info_lines)
        self.logger.info(f"\nTensorRT Engine Input/Output Information:\n{io_info_str}")

    def infer_mg_cpu(self, model, input_data):
        # Ensure model and input data are on CPU
        input_data = input_data.cpu()
        model = model.cpu()

        # Start timing CPU operations
        start_time = time.time()

        # INFERENCE!
        preds = model(input_data)

        # End timing CPU operations
        end_time = time.time()

        # Calculate latency
        latency = (
            end_time - start_time
        ) * 1000.0  # Convert from seconds to milliseconds

        # Return the predictions
        return preds.detach(), latency

    def infer_mg_cuda(self, model, input_data):
        # send model and input data to GPU for inference
        input_data = input_data.cuda()
        model = model.cuda()

        # Create CUDA events for timing GPU operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # INFERENCE!
        start.record()
        preds = model(input_data)
        end.record()

        # Synchronize to ensure all GPU operations are finished
        torch.cuda.synchronize()

        # Calculate latency between start and end events
        latency = start.elapsed_time(end)

        # return the prediction back to the CPU
        return preds.detach().cpu(), latency

    def infer_trt_cuda(self, trt_context, input_data):
        bufferH = []
        bufferH.append(np.ascontiguousarray(input_data))
        for i in range(self.n_Input, self.num_io):
            bufferH.append(
                np.empty(
                    self.context.get_tensor_shape(self.lTensorName[i]),
                    dtype=trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i])),
                )
            )
        bufferD = []
        for i in range(self.num_io):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(self.n_Input):
            cudart.cudaMemcpy(
                bufferD[i],
                bufferH[i].ctypes.data,
                bufferH[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )

        for i in range(self.num_io):
            self.context.set_tensor_address(self.lTensorName[i], int(bufferD[i]))

        # Create CUDA events for timing GPU operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # INFERENCE!
        start.record()
        self.context.execute_async_v3(0)
        end.record()

        # Synchronize to ensure all GPU operations are finished
        torch.cuda.synchronize()

        # Calculate latency between start and end events
        latency = start.elapsed_time(end)

        # Copying data from device to host and collecting output tensors
        output_data = [
            bufferH[i]
            for i in range(self.n_Input, self.num_io)
            for _ in [
                cudart.cudaMemcpy(
                    bufferH[i].ctypes.data,
                    bufferD[i],
                    bufferH[i].nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
            ]
        ]

        # Flatten output if it consists of only one item
        output_data = output_data[0] if len(output_data) == 1 else output_data

        for b in bufferD:
            cudart.cudaFree(b)

        # Convert the raw scores from numpy array to PyTorch tensor
        preds_tensor = torch.tensor(output_data, device="cpu", dtype=torch.float32)

        return preds_tensor, latency

    def infer_onnx_cpu(self, ort_inference_session, input_data):
        # Convert PyTorch tensor to numpy array for ONNX Runtime
        input_data_np = input_data.numpy()

        # Start timing CPU operations
        start_time = time.time()

        # Run inference using ONNX Runtime
        output_data = ort_inference_session.run(None, {"input": input_data_np})

        # End timing CPU operations
        end_time = time.time()

        # Calculate latency in milliseconds
        latency = (
            end_time - start_time
        ) * 1000.0  # Convert from seconds to milliseconds

        # Flatten output if it consists of only one item
        output_data = output_data[0] if len(output_data) == 1 else output_data

        # Convert the raw scores from numpy array back to PyTorch tensor
        preds_tensor = torch.from_numpy(
            output_data
        ).float()  # Ensure tensor is on CPU and in float32 format

        return preds_tensor, latency

    def infer_onnx_cuda(self, ort_inference_session, input_data):
        input_data_np = input_data.numpy()

        # Create CUDA events for timing GPU operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output_data = ort_inference_session.run(None, {"input": input_data_np})
        end.record()

        # Synchronize to ensure all GPU operations are finished
        torch.cuda.synchronize()

        # Calculate latency between start and end events
        latency = start.elapsed_time(end)

        # Flatten output if it consists of only one item
        output_data = output_data[0] if len(output_data) == 1 else output_data

        # Convert the raw scores from numpy array to PyTorch tensor
        preds_tensor = torch.tensor(output_data, device="cpu", dtype=torch.float32)

        return preds_tensor, latency

    def evaluate(self):
        self.logger.info(f"Starting transformation analysis on {self.model_name}")

        num_GPU_warmup_batches = self.config["num_GPU_warmup_batches"]

        match self.config["task"]:
            case "cls":
                metric = torchmetrics.classification.MulticlassAccuracy(
                    num_classes=self.num_of_classes
                )
                precision_metric = torchmetrics.Precision(
                    num_classes=self.num_of_classes,
                    average="weighted",
                    task="multiclass",
                )
                recall_metric = torchmetrics.Recall(
                    num_classes=self.num_of_classes,
                    average="weighted",
                    task="multiclass",
                )
                f1_metric = torchmetrics.F1Score(
                    num_classes=self.num_of_classes,
                    average="weighted",
                    task="multiclass",
                )
            case _:
                raise Exception(
                    f"Unsupported task type {self.config['task']}. Please set a supported task type in the config file."
                )

        # Initialize lists to store metrics for each configuration
        recorded_accs = []
        latencies = []
        gpu_power_usages = []
        accs, losses = [], []

        if "test" in self.config and self.config["test"]:
            dataloader = self.config["data_module"].test_dataloader()
            dataset = "Test"
        else:
            dataloader = self.config["data_module"].val_dataloader()
            dataset = "Validation"

        # Iterate over batches in the validation/train dataset
        for j, (xs, ys) in enumerate(dataloader):
            # Break the loop after processing the specified number of batches or to drop the last incomplete batch
            if (
                j >= self.config["num_batches"]
                or xs.shape[0] != self.config["batch_size"]
            ):
                break

            # Instantiate and start the power monitor
            power_monitor = PowerMonitor(self.config)
            power_monitor.start()

            # Synchronize to ensure all GPU operations are finished
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # TRT Inference
            if isinstance(self.model, trt.IExecutionContext):
                if self.config["accelerator"] != "cuda":
                    raise Exception(
                        "TensorRT inference is only supported on CUDA devices."
                    )
                preds, latency = self.infer_trt_cuda(self.model, xs)

            # ONNX Inference
            elif isinstance(self.model, ort.InferenceSession):
                if self.config["accelerator"] == "cpu":
                    preds, latency = self.infer_onnx_cpu(self.model, xs)
                elif self.config["accelerator"] == "cuda":
                    preds, latency = self.infer_onnx_cuda(self.model, xs)
                else:
                    raise Exception(
                        f"ONNX inference is not support by device {self.config['accelerator']}."
                    )

            # MaseGraph Inference
            else:
                if self.config["accelerator"] == "cpu":
                    preds, latency = self.infer_mg_cpu(self.model, xs)
                elif self.config["accelerator"] == "cuda":
                    preds, latency = self.infer_mg_cuda(self.model, xs)
                else:
                    raise Exception(
                        f"MaseGraph inference is not support by device {self.config['accelerator']}."
                    )

            # Stop the power monitor and calculate average power
            power_monitor.stop()
            power_monitor.join()  # Ensure monitoring thread has finished

            # Skip the first number of iterations to allow the model to warm up
            if j < num_GPU_warmup_batches:
                continue

            latencies.append(latency)

            avg_power = (
                sum(power_monitor.power_readings) / len(power_monitor.power_readings)
                if power_monitor.power_readings
                else 0
            )
            # Store the calculated average power
            gpu_power_usages.append(avg_power)

            # Calculate accuracy and loss for the batch
            loss = torch.nn.functional.cross_entropy(preds, ys)
            acc = metric(preds, ys)
            accs.append(acc.item())
            losses.append(loss.item())

            # Update torchmetrics metrics
            preds_labels = torch.argmax(preds, dim=1)
            precision_metric(preds_labels, ys)
            recall_metric(preds_labels, ys)
            f1_metric(preds_labels, ys)

        # Compute final precision, recall, and F1 for this configuration
        avg_precision = precision_metric.compute()
        avg_recall = recall_metric.compute()
        avg_f1 = f1_metric.compute()

        # Convert metrics to float if they are tensors
        avg_precision = (
            avg_precision.item() if torch.is_tensor(avg_precision) else avg_precision
        )
        avg_recall = avg_recall.item() if torch.is_tensor(avg_recall) else avg_recall
        avg_f1 = avg_f1.item() if torch.is_tensor(avg_f1) else avg_f1

        # Reset metrics for the next configuration
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()

        # Calculate and record average metrics for the current configuration
        acc_avg = sum(accs) / len(accs)
        loss_avg = sum(losses) / len(losses)
        recorded_accs.append(acc_avg)
        avg_latency = sum(latencies) / len(latencies)
        avg_gpu_power_usage = sum(gpu_power_usages) / len(gpu_power_usages)
        avg_gpu_energy_usage = (avg_gpu_power_usage * 1000) * (avg_latency / 3600000)

        metrics = [
            ["Average " + dataset + " Accuracy", f"{acc_avg:.5g}"],
            ["Average Precision", f"{avg_precision:.5g}"],
            ["Average Recall", f"{avg_recall:.5g}"],
            ["Average F1 Score", f"{avg_f1:.5g}"],
            ["Average Loss", f"{loss_avg:.5g}"],
            ["Average Latency", f"{avg_latency:.5g} ms"],
            ["Average GPU Power Usage", f"{avg_gpu_power_usage:.5g} W"],
            ["Inference Energy Consumption", f"{avg_gpu_energy_usage:.5g} mWh"],
        ]

        # Formatting the table with tabulate
        formatted_metrics = tabulate(
            metrics,
            headers=["Metric (Per Batch)", "Value"],
            tablefmt="pretty",
            floatfmt=".5g",
        )

        # Print result summary
        self.logger.info(f"\nResults {self.model_name}:\n" + formatted_metrics)

        # Store the results in a dictionary and return it
        results = {
            "Average Accuracy": acc_avg,
            "Average Precision": avg_precision,
            "Average Recall": avg_recall,
            "Average F1 Score": avg_f1,
            "Average Loss": loss_avg,
            "Average Latency": avg_latency,
            "Average GPU Power Usage": avg_gpu_power_usage,
            "Inference Energy Consumption": avg_gpu_energy_usage,
        }
        return results
