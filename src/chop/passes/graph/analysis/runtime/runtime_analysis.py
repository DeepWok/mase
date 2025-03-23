import logging
import torch
from pathlib import PosixPath
from chop.ir import MaseGraph
from .utils import PowerMonitor, get_execution_provider
import os
import inspect
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
    Evaluates the performance of a model by analyzing its inference speed, accuracy, and other
    relevant metrics (e.g., Word Error Rate for CTC tasks). This function is part of the model
    optimization and evaluation pipeline, showcasing improvements in inference speed achieved
    through quantization, graph-based optimizations, and potential conversion to TensorRT or ONNX
    formats.

    It accepts models in various formats:
      - A `MaseGraph` object (e.g., a graph-transformed PyTorch model),
      - A `PosixPath` to an ONNX model (e.g., "model.onnx"),
      - A `PosixPath` to a TensorRT engine (e.g., "model.trt").

    Depending on the `task` specified (e.g., "ctc" for speech recognition, "cls" for classification),
    the function computes different performance metrics:

      - **CTC tasks** (e.g., speech recognition):
        - Word Error Rate (WER)
        - Real-Time Factor (RTF) — a measure of latency relative to audio duration

      - **Classification tasks**:
        - Accuracy, Precision, Recall, F1 Score
        - Cross-Entropy Loss

    In addition, the function tracks:
      - **Latency** (ms)
      - **GPU Power Usage** (Watts)
      - **Inference Energy Consumption** (mWh)

    The analysis is conducted by creating an instance of `RuntimeAnalysis` with the model and
    optional arguments, evaluating the model's performance, and then storing the results. Users
    can pass arguments such as `decoder`, `beam_width`, `ctc_head`, or `tokenizer` for CTC tasks
    to ensure correct decoding (including attention_mask handling). For classification tasks,
    metrics like Accuracy, Precision, Recall, and F1 are automatically computed.

    :param model: 
        The model to be analyzed. Can be a `MaseGraph` instance or a `PosixPath` to either
        an ONNX model or a TensorRT engine.
    :type model: MaseGraph or PosixPath

    :param pass_args: 
        Optional dictionary of arguments that may influence the analysis:
        - "task": Either "ctc" (speech recognition) or "cls" (classification).
        - "decoder": Optional decoder object for beam or greedy decoding in CTC tasks.
        - "beam_width": Beam size for CTC decoding.
        - "ctc_head": CTC output head (torch.nn.Module) used after the encoder for speech tasks.
        - "tokenizer": Tokenizer for decoding predictions to text.
        - "batch_size": Batch size for evaluation.
        - "num_batches": Number of batches to evaluate.
        - "accelerator": Device for inference ("cpu" or "cuda").
        - "data_module": Provides dataloaders for the dataset.
        - Additional configuration parameters (e.g., power monitoring).

    :type pass_args: dict, optional

    :return: 
        A tuple containing:
        1) The original model (unmodified),
        2) A dictionary with the results of the analysis, including metrics such as:
           - Average WER (for CTC tasks),
           - Average Latency (ms),
           - Average RTF (for CTC tasks),
           - Average GPU Power Usage (W),
           - Inference Energy Consumption (mWh),
           - (If classification) Accuracy, Precision, Recall, F1 Score, and Loss.

    :rtype: tuple(MaseGraph or PosixPath, dict)

    **Example usage**::

        from pathlib import PosixPath

        # Evaluate a TensorRT engine
        model_path = PosixPath('/path/to/model.trt')
        _, results = runtime_analysis_pass(model_path, pass_args={
            "task": "ctc",
            "decoder": my_ctc_decoder,
            "beam_width": 10,
            "ctc_head": my_ctc_head,
            "tokenizer": my_tokenizer,
            "accelerator": "cuda",
            "batch_size": 4,
            ...
        })

        print(results)  # Dictionary of performance metrics

    This function provides a holistic view of the model's operational characteristics, enabling a
    thorough comparison between the original unquantized model, various quantized versions (INT8,
    for example), and other optimizations. Metrics like WER (or Accuracy) and latency/power usage
    provide valuable insights into the trade-offs between model size, speed, and accuracy, guiding
    informed decisions about deployment in production environments.
    """
    try:
        import tensorrt as trt # type: ignore
        globals()['trt'] = trt
    except ImportError:
        raise ImportError("tensorrt is required for this functionality. Please install it from NVIDIA's repositories.")

    try:
        from cuda import cudart # type: ignore
        globals()['cudart'] = cudart
    except ImportError:
        raise ImportError("pycuda's cudart is required for this functionality. Please install pycuda.")

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

    def infer_mg_cpu(self, model, input_values, attention_mask=None):
        # Ensure model and input data are on CPU
        input_values = input_values.cpu()
        model = model.cpu()

        if attention_mask is not None:
            attention_mask = attention_mask.cpu()
            inputs = (input_values, attention_mask)
        else:
            inputs = (input_values,)
            
        # Start timing CPU operations
        start_time = time.time()

        # INFERENCE!
        preds = model(*inputs)

        # End timing CPU operations
        end_time = time.time()

        # Calculate latency
        latency = (
            end_time - start_time
        ) * 1000.0  # Convert from seconds to milliseconds

        if isinstance(preds, dict):
            detached_dict = {}
            for k, v in preds.items():
                if torch.is_tensor(v):
                    detached_dict[k] = v.detach()
                else:
                    detached_dict[k] = v
            preds = detached_dict
        elif torch.is_tensor(preds):
            preds = preds.detach()

        return preds, latency

    def infer_mg_cuda(self, model, input_values, attention_mask=None):
        # send model and input data to GPU for inference
        input_values = input_values.cuda()
        model = model.cuda()

        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
            inputs = (input_values, attention_mask)
        else:
            inputs = (input_values,)

        # Create CUDA events for timing GPU operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # INFERENCE!
        start.record()
        preds = model(*inputs)
        end.record()

        # Synchronize to ensure all GPU operations are finished
        torch.cuda.synchronize()

        # Calculate latency between start and end events
        latency = start.elapsed_time(end)

        if isinstance(preds, dict):
            detached_dict = {}
            for k, v in preds.items():
                detached_dict[k] = v.detach().cpu() if torch.is_tensor(v) else v
            preds = detached_dict
        elif torch.is_tensor(preds):
            preds = preds.detach().cpu()

        return preds, latency

    def infer_trt_cuda(self, trt_context, input_values, attention_mask=None):
        input_values_np = input_values.cpu().numpy()  # ensure on CPU before converting
        if attention_mask is not None:
            attention_mask_np = attention_mask.cpu().numpy()
            input_arrays = [input_values_np, attention_mask_np]
        else:
            input_arrays = [input_values_np]
            
        bufferH = []
        for arr in input_arrays:
            bufferH.append(np.ascontiguousarray(arr))

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
        output_data = []
        for i in range(self.n_Input, self.num_io):
            cudart.cudaMemcpy(
                bufferH[i].ctypes.data,
                bufferD[i],
                bufferH[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
            output_data.append(bufferH[i])

        # Flatten output if it consists of only one item
        output_data = output_data[0] if len(output_data) == 1 else output_data

        for b in bufferD:
            cudart.cudaFree(b)

        if len(output_data) == 1:
            output_data = output_data[0]
        # Convert the raw scores from numpy array to PyTorch tensor
        preds_tensor = torch.tensor(output_data, device="cpu", dtype=torch.float32)

        return preds_tensor, latency

    def infer_onnx_cpu(self, ort_inference_session, input_values, attention_mask=None):
        # Convert PyTorch tensor to numpy array for ONNX Runtime
        input_values_np = input_values.cpu().numpy()
        if attention_mask is not None:
            attention_mask_np = attention_mask.cpu().numpy()
            inputs = {"input": {"input_values": input_values_np, "attention_mask": attention_mask_np}}
        else:
            inputs = {"input": {"input_values": input_values_np}}

        # Run inference using ONNX Runtime
        start_time = time.time()
        output_data = ort_inference_session.run(None, inputs)

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

    def infer_onnx_cuda(self, ort_inference_session, input_values, attention_mask=None):
        input_values_np = input_values.cpu().numpy()

        if attention_mask is not None:
            attention_mask_np = attention_mask.cpu().numpy()
            inputs = {"input": {"input_values": input_values_np, "attention_mask": attention_mask_np}}
        else:
            inputs = {"input": {"input_values": input_values_np}}
        
        # Create CUDA events for timing GPU operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output_data = ort_inference_session.run(None, inputs)
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
        task = self.config["task"]

        requires_attention_mask = self.config.get(
            "requires_attention_mask",
            "attention_mask" in inspect.signature(self.model.forward).parameters
        )        

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
            ctc_head = self.config.get("ctc_head", None)
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
        for j, batch in enumerate(dataloader):
            xs = batch["input_values"]
            ys = batch["labels"]
            attention_mask = batch["attention_mask"]

            # Stop if we've exceeded our number of evaluation batches, or if the batch is incomplete
            if j >= self.config["num_batches"] or xs.shape[0] != self.config["batch_size"]:
                print(f"[DEBUG] Batch {j+1} - xs shape: {xs.shape}, Expected batch size: {self.config['batch_size']}")
                break

            # Power monitoring (start)
            power_monitor = PowerMonitor(self.config)
            power_monitor.start()

            # Clear GPU caches & sync
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            if requires_attention_mask:
                attention_mask = attention_mask.to(self.config["accelerator"])
                model_input = (xs, attention_mask)
            else:
                model_input = (xs,)

            # ---------------- (A) RUN INFERENCE (TRT, ONNX, or MaseGraph) ----------------
            if isinstance(self.model, trt.IExecutionContext):
                # TensorRT
                if self.config["accelerator"] != "cuda":
                    raise Exception("TensorRT inference is only supported on CUDA devices.")
                preds, latency = self.infer_trt_cuda(self.model, *model_input)

            elif isinstance(self.model, ort.InferenceSession):
                # ONNX Runtime
                if self.config["accelerator"] == "cpu":
                    preds, latency = self.infer_onnx_cpu(self.model, *model_input)
                elif self.config["accelerator"] == "cuda":
                    preds, latency = self.infer_onnx_cuda(self.model, *model_input)
                else:
                    raise Exception(
                        f"ONNX inference is not support by device {self.config['accelerator']}."
                    )

            else:
                # MaseGraph or raw PyTorch
                if self.config["accelerator"] == "cpu":
                    preds, latency = self.infer_mg_cpu(self.model, *model_input)
                elif self.config["accelerator"] == "cuda":
                    print(f"[DEBUG] Running MaseGraph inference on batch {j+1}")
                    preds, latency = self.infer_mg_cuda(self.model, *model_input)
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
                # Real-Time Factor (RTF) = Latency / (1 / sample_rate)
                max_length = xs.shape[1]
                audio_duration = max_length / self.config["sample_rate"]
                rtf = (latency / 1000.0) / audio_duration
                rtfs.append(rtf)

                # WER CTC logic
                # preds: [batch_size, time_steps, vocab_size] or [batch_size, *]
                # ys: [batch_size, time_steps]

                # Apply CTC decoding to get predicted text
                if ctc_head is None:
                    raise Exception("CTC head must be provided in config for full model evaluation")
                
                ctc_head = ctc_head.cpu()
                encoder_output = preds["last_hidden_state"].cpu()

                predictions = ctc_head(encoder_output)

                if torch.is_tensor(predictions):
                    predictions = predictions.detach().cpu()
                else:
                    predictions = torch.tensor(predictions).cpu()

                preds_np = predictions.numpy()

                pred_texts = []
                label_texts = []

                for i in range(preds_np.shape[0]):
                    sample_logits = torch.from_numpy(preds_np[i])
                    sample_log_probs = sample_logits.log_softmax(dim=-1).cpu().numpy()
                    if decoder is not None:
                        transcription = decoder.decode(sample_log_probs, beam_width=beam_width)
                    else:
                        raise Exception(
                            "Decoder must be provided for CTC runtime analysis. Pass 'decoder' in config.")
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
            avg_rtf = sum(rtfs) / len(rtfs) if rtfs else 0

            # CTC final metrics
            avg_wer = sum(batch_wers) / len(batch_wers) if batch_wers else 0

            metrics_table = [
                ["Average " + dataset + " WER", f"{avg_wer:.5g}"],
                ["Average Latency", f"{avg_latency:.5g} ms"],
                ["Average RTF", f"{avg_rtf:.5g}"],
                ["Average GPU Power Usage", f"{avg_gpu_power_usage:.5g} W"],
                ["Inference Energy Consumption", f"{avg_gpu_energy_usage:.5g} mWh"],
            ]

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
