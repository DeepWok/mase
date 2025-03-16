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


from chop.passes.utils import register_mase_pass


@register_mase_pass(
    "runtime_analysis_pass",
    dependencies=[
        "pytorch_quantization",
        "tensorrt",
        "pynvml",
        "pycuda",
        "cuda",
    ],
)
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
    import tensorrt as trt
    from cuda import cudart

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
        model_dir = f'{self.config["model"]}_{self.config["task"]}_{self.config["data_module"].name}_{current_date}'
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

        print(f"[DEBUG] ONNX Inference - Input Shape: {input_data_np.shape}")
        start_time = time.time()

        # Run inference using ONNX Runtime
        output_data = ort_inference_session.run(None, {"input": input_data_np})

        # End timing CPU operations
        end_time = time.time()

        print(f"[DEBUG] ONNX Inference - Output Shape: {output_data[0].shape}")
        print(f"[DEBUG] ONNX Inference - Latency: {(end_time - start_time) * 1000:.3f} ms")
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

        print(f"[DEBUG] ONNX Inference - Input Shape: {input_data_np.shape}")

        # Create CUDA events for timing GPU operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output_data = ort_inference_session.run(None, {"input": input_data_np})
        end.record()

        print(f"[DEBUG] ONNX Inference - Output Shape: {output_data[0].shape}")
        print(f"[DEBUG] ONNX Inference - Latency: {(end_time - start_time) * 1000:.3f} ms")

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

        print(f"[DEBUG] Dataset Type: {self.config['data_module'].name}")
        print(f"[DEBUG] Batch Size: {self.config['batch_size']}")
        print(f"[DEBUG] Dataloader Length: {len(dataloader)}")

        self.logger.info(f"Starting transformation analysis on {self.model_name}")

        num_GPU_warmup_batches = self.config["num_GPU_warmup_batches"]
        task = self.config["task"]

        # ---------- 1) SET UP METRICS BASED ON TASK ----------
        if task == "cls":
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

            # Arrays to store classification metrics
            accs, losses = [], []

        elif task == "ctc":
            wer_metric = torchmetrics.text.WordErrorRate()
            decoder = self.config.get("decoder", None)
            beam_width = self.config.get("beam_width", 10)
            tokenizer = self.config["tokenizer"]
            padding_value = self.config.get("padding_value", -100)

            # We'll collect WER from each batch
            batch_wers = []

        else:
            raise Exception(
                f"Unsupported task type {task}. Please set a supported task type in the config file."
            )
    
        # ---------- 2) PREPARE DATA LOADER (TEST OR VALIDATION) ----------
        if "test" in self.config and self.config["test"]:
            dataloader = self.config["data_module"].test_dataloader()
            dataset = "Test"
        else:
            dataloader = self.config["data_module"].val_dataloader()
            dataset = "Validation"

        # ---------- 3) ARRAYS FOR LATENCIES & POWER FOR ALL TASKS ----------
        latencies = []
        gpu_power_usages = []
        rtfs = [] 

        # ---------- 4) MAIN EVALUATION LOOP ----------
        for j, (xs, ys) in enumerate(dataloader):
            # Stop if we've exceeded our number of evaluation batches, or if the batch is incomplete
            if (
                j >= self.config["num_batches"]
                or xs.shape[0] != self.config["batch_size"]
            ):
                break

            print(f"[DEBUG] Running Batch {j+1}/{self.config['num_batches']}")

            # Power monitoring (start)
            power_monitor = PowerMonitor(self.config)
            power_monitor.start()

            # Clear GPU caches & sync
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            if isinstance(self.model, ort.InferenceSession):
                print(f"[DEBUG] Running ONNX inference on batch {j+1}")
            elif isinstance(self.model, trt.IExecutionContext):
                print(f"[DEBUG] Running TensorRT inference on batch {j+1}")
            else:
                print(f"[DEBUG] Running MaseGraph inference on batch {j+1}")

            # ---------------- (A) RUN INFERENCE (TRT, ONNX, or MaseGraph) ----------------
            if isinstance(self.model, trt.IExecutionContext):
                # TensorRT
                if self.config["accelerator"] != "cuda":
                    raise Exception("TensorRT inference is only supported on CUDA devices.")
                preds, latency = self.infer_trt_cuda(self.model, xs)

            elif isinstance(self.model, ort.InferenceSession):
                # ONNX Runtime
                if self.config["accelerator"] == "cpu":
                    preds, latency = self.infer_onnx_cpu(self.model, xs)
                elif self.config["accelerator"] == "cuda":
                    preds, latency = self.infer_onnx_cuda(self.model, xs)
                else:
                    raise Exception(
                        f"ONNX inference is not support by device {self.config['accelerator']}."
                    )

            else:
                # MaseGraph or raw PyTorch
                if self.config["accelerator"] == "cpu":
                    preds, latency = self.infer_mg_cpu(self.model, xs)
                elif self.config["accelerator"] == "cuda":
                    preds, latency = self.infer_mg_cuda(self.model, xs)
                else:
                    raise Exception(
                        f"MaseGraph inference is not support by device {self.config['accelerator']}."
                    )

            # Power monitoring (stop)
            power_monitor.stop()
            power_monitor.join()

            # Skip warmup batches
            if j < num_GPU_warmup_batches:
                continue

            # Record latency
            latencies.append(latency)

            # Compute average power usage for this batch
            avg_power = (
                sum(power_monitor.power_readings) / len(power_monitor.power_readings)
                if power_monitor.power_readings
                else 0
            )
            gpu_power_usages.append(avg_power)

            print(f"[DEBUG] Batch {j+1} - Latency: {latency:.3f} ms")
            print(f"[DEBUG] Batch {j+1} - GPU Power Usage: {avg_power:.3f} W")
            print(f"[DEBUG] Batch {j+1} - RTF: {rtf:.3f}")

            # ---------- (B) METRICS DEPENDING ON TASK ----------
            if task == "cls":
                # Classification logic
                # preds: [batch_size, num_classes]
                # ys: [batch_size]

                # Cross-entropy (classification) loss
                loss = torch.nn.functional.cross_entropy(preds, ys)
                losses.append(loss.item())

                # Accuracy (MulticlassAccuracy)
                acc = metric(preds, ys)
                accs.append(acc.item())

                # Update precision, recall, F1
                preds_labels = torch.argmax(preds, dim=1)
                precision_metric(preds_labels, ys)
                recall_metric(preds_labels, ys)
                f1_metric(preds_labels, ys)

            elif task == "ctc":
                print(f"[DEBUG] Pred Texts: {pred_texts}")
                print(f"[DEBUG] Label Texts: {label_texts}")

                if not pred_texts or not label_texts:
                    print("[ERROR] Empty prediction or label text! Check tokenizer or decoder.")

                # Real-Time Factor (RTF) = Latency / (1 / sample_rate)
                max_length = xs.shape[1]
                audio_duration = max_length / self.config["sample_rate"]
                rtf = (latency / 1000.0) / audio_duration
                rtfs.append(rtf)

                # WER CTC logic
                # preds: [batch_size, time_steps, vocab_size] or [batch_size, *]
                # ys: [batch_size, time_steps]

                # Convert preds to numpy if needed
                if torch.is_tensor(preds):
                    # shape: (batch_size, time_steps, vocab_size)
                    preds = preds.cpu().numpy()

                # We'll decode each batch => get WER
                pred_texts = []
                label_texts = []

                for i in range(preds.shape[0]):
                    sample_logits = torch.from_numpy(preds[i])
                    sample_log_probs = sample_logits.log_softmax(dim=-1).cpu().numpy()

                    if decoder is not None:
                        transcription = decoder.decode(sample_log_probs, beam_width=beam_width)
                    else:
                        raise Exception(
                            "Decoder must be provided for CTC runtime analysis. Pass 'decoder' in config."
                        )

                    pred_texts.append(transcription.lower())

                for label_seq in ys:
                    label_filtered = [token for token in label_seq if token != padding_value]
                    label_text = tokenizer.decode(label_filtered, skip_special_tokens=True)
                    label_texts.append(label_text.lower())

                # Now compute batch WER
                batch_wer = wer_metric(pred_texts, label_texts)
                batch_wers.append(batch_wer.item() if torch.is_tensor(batch_wer) else batch_wer)

        # ---------- 5) AFTER LOOP, COMPUTE FINAL METRICS BASED ON TASK ----------
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        avg_gpu_power_usage = sum(gpu_power_usages) / len(gpu_power_usages) if gpu_power_usages else 0
        avg_gpu_energy_usage = (avg_gpu_power_usage * 1000) * (avg_latency / 3600000)
        avg_rtf = sum(rtfs) / len(rtfs) if rtfs else 0

        if task == "cls":
            # Classification final metrics
            avg_acc = sum(accs) / len(accs) if accs else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            avg_precision = precision_metric.compute()
            avg_recall = recall_metric.compute()
            avg_f1 = f1_metric.compute()

            # Convert to float
            avg_precision = avg_precision.item() if torch.is_tensor(avg_precision) else avg_precision
            avg_recall = avg_recall.item() if torch.is_tensor(avg_recall) else avg_recall
            avg_f1 = avg_f1.item() if torch.is_tensor(avg_f1) else avg_f1

            # Reset metrics
            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()

            metrics_table = [
                ["Average " + dataset + " Accuracy", f"{avg_acc:.5g}"],
                ["Average Precision", f"{avg_precision:.5g}"],
                ["Average Recall", f"{avg_recall:.5g}"],
                ["Average F1 Score", f"{avg_f1:.5g}"],
                ["Average Loss", f"{avg_loss:.5g}"],
                ["Average Latency", f"{avg_latency:.5g} ms"],
                ["Average GPU Power Usage", f"{avg_gpu_power_usage:.5g} W"],
                ["Inference Energy Consumption", f"{avg_gpu_energy_usage:.5g} mWh"],
            ]

            results = {
                "Average Accuracy": avg_acc,
                "Average Precision": avg_precision,
                "Average Recall": avg_recall,
                "Average F1 Score": avg_f1,
                "Average Loss": avg_loss,
                "Average Latency": avg_latency,
                "Average GPU Power Usage": avg_gpu_power_usage,
                "Inference Energy Consumption": avg_gpu_energy_usage,
            }

        elif task == "ctc":
            # Real-Time Factor metric
            avg_rtf = sum(latencies) / len(latencies) if latencies else 0

            # CTC final metrics
            avg_wer = sum(batch_wers) / len(batch_wers) if batch_wers else 0

            metrics_table = [
                ["Average " + dataset + " WER", f"{avg_wer:.5g}"],
                ["Average Latency", f"{avg_latency:.5g} ms"],
                ["Average RTF", f"{avg_rtf:.5g}"],
                ["Average GPU Power Usage", f"{avg_gpu_power_usage:.5g} W"],
                ["Inference Energy Consumption", f"{avg_gpu_energy_usage:.5g} mWh"],
            ]

            print(f"[DEBUG] Average Latency: {avg_latency:.3f} ms")
            print(f"[DEBUG] Average GPU Power Usage: {avg_gpu_power_usage:.3f} W")
            print(f"[DEBUG] Average RTF: {avg_rtf:.3f}")
            print(f"[DEBUG] Average WER: {avg_wer:.3f}")
            print(f"[DEBUG] Inference Energy Consumption: {avg_gpu_energy_usage:.3f} mWh")

            results = {
                "Average WER": avg_wer,
                "Average Latency": avg_latency,
                "Average RTF": avg_rtf,
                "Average GPU Power Usage": avg_gpu_power_usage,
                "Inference Energy Consumption": avg_gpu_energy_usage,
            }

        # ---------- 6) LOG RESULTS ----------
        formatted_metrics = tabulate(
            metrics_table,
            headers=["Metric (Per Batch)", "Value"],
            tablefmt="pretty",
            floatfmt=".5g",
        )
        self.logger.info(f"\nResults {self.model_name}:\n" + formatted_metrics)

        return results
