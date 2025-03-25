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
import jiwer

from chop.passes.utils import register_mase_pass

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
    """Runtime analysis pass"""
    print("[DEBUG] Starting runtime_analysis_pass")
    try:
        import tensorrt as trt # type: ignore
        globals()['trt'] = trt
        print("[DEBUG] Successfully imported tensorrt")
    except ImportError as e:
        print(f"[DEBUG] Failed to import tensorrt: {e}")
        raise ImportError("tensorrt is required for this functionality. Please install it from NVIDIA's repositories.")

    try:
        from cuda import cudart # type: ignore
        globals()['cudart'] = cudart
        print("[DEBUG] Successfully imported cuda.cudart")
    except ImportError as e:
        print(f"[DEBUG] Failed to import cuda.cudart: {e}")
        raise ImportError("pycuda's cudart is required for this functionality. Please install pycuda.")

    print(f"[DEBUG] Creating RuntimeAnalysis with model type: {type(model)}")
    analysis = RuntimeAnalysis(model, pass_args)
    print("[DEBUG] Starting evaluation")
    results = analysis.evaluate()
    print(f"[DEBUG] Evaluation completed. Results: {results}")
    analysis.store(results)

    return model, results


class RuntimeAnalysis:
    def __init__(self, model, config):
        print(f"[DEBUG] Initializing RuntimeAnalysis with config: {config.keys()}")
        
        # Instantiate default performance analyzer args
        if "num_batches" not in config.keys():
            config["num_batches"] = 500
            config["num_GPU_warmup_batches"] = 5
            config["test"] = True
            print("[DEBUG] Added default config values")

        self.config = config

        self.logger = logging.getLogger(__name__)
        print(f"[DEBUG] Data module info: {self.config.get('data_module', None)}")
        self.num_of_classes = self.config["data_module"].dataset_info.num_classes
        print(f"[DEBUG] Number of classes: {self.num_of_classes}")

        print(f"[DEBUG] Processing model of type: {type(model)}")
        match model:
            case MaseGraph():
                # Check if model is mase graph
                self.model = model.model
                self.model_name = self.config["model"]
                self.model_type = "mase_graph"
                print(f"[DEBUG] Model is MaseGraph: {self.model_name}")

            case PosixPath() as path:
                print(f"[DEBUG] Model is PosixPath: {path}")
                match path.suffix:
                    case ".trt":
                        print("[DEBUG] Loading TensorRT engine")
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
                        print(f"[DEBUG] TensorRT engine loaded with {self.num_io} tensors")

                    case ".onnx":
                        print("[DEBUG] Loading ONNX model")
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
                        print(f"[DEBUG] ONNX model loaded: {self.model_name}")
                    case _:
                        # If file type is neither .trt nor .onnx
                        print(f"[DEBUG] Unsupported file type: {path.suffix}")
                        raise Exception(
                            "Model must be a MaseGraph or a path to a trt file. Have you run the quantization pass?"
                        )
            case _:
                # If model is neither MaseGraph nor PosixPath
                print(f"[DEBUG] Unsupported model type: {type(model)}")
                raise Exception(
                    "Model must be a MaseGraph or a PosixPath to a trt file. Have you run the quantization pass?"
                )

    def store(self, results):
        # Save the results in a JSON file
        print(f"[DEBUG] Storing results: {results}")
        save_path = self._prepare_save_path(self.model_type, "json")
        with open(save_path, "w") as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Runtime analysis results saved to {save_path}")
        print(f"[DEBUG] Results saved to {save_path}")

    def _prepare_save_path(self, method: str, suffix: str):
        """Creates and returns a save path for the model."""
        print("[DEBUG] Preparing save path")
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
        full_path = save_dir / f"model.{suffix}"
        print(f"[DEBUG] Save path created: {full_path}")
        return full_path

    def _summarize(self):
        print("[DEBUG] Summarizing TensorRT engine")
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
        print(f"[DEBUG] TensorRT engine summary completed")

    def infer_mg_cpu(self, model, input_values, attention_mask=None):
        print(f"[DEBUG] Starting MaseGraph CPU inference with input shape: {input_values.shape}")
        # Ensure model and input data are on CPU
        input_values = input_values.cpu()
        model = model.cpu()

        if attention_mask is not None:
            print(f"[DEBUG] Using attention_mask with shape: {attention_mask.shape}")
            attention_mask = attention_mask.cpu()
            inputs = (input_values, attention_mask)
        else:
            print("[DEBUG] No attention_mask provided")
            inputs = (input_values,)
            
        # Start timing CPU operations
        start_time = time.time()

        # INFERENCE!
        print("[DEBUG] Running model inference on CPU")
        try:
            preds = model(*inputs)
            print(f"[DEBUG] Inference successful. Output type: {type(preds)}")
            if isinstance(preds, dict):
                print(f"[DEBUG] Prediction keys: {preds.keys()}")
                for k, v in preds.items():
                    if torch.is_tensor(v):
                        print(f"[DEBUG] Prediction [{k}] shape: {v.shape}, dtype: {v.dtype}")
                    else:
                        print(f"[DEBUG] Prediction [{k}] type: {type(v)}")
            elif torch.is_tensor(preds):
                print(f"[DEBUG] Prediction tensor shape: {preds.shape}, dtype: {preds.dtype}")
        except Exception as e:
            print(f"[DEBUG] Error during model inference: {e}")
            raise

        # End timing CPU operations
        end_time = time.time()

        # Calculate latency
        latency = (
            end_time - start_time
        ) * 1000.0  # Convert from seconds to milliseconds
        print(f"[DEBUG] Inference latency: {latency:.2f} ms")

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
        print(f"[DEBUG] Starting MaseGraph CUDA inference with input shape: {input_values.shape}")
        # send model and input data to GPU for inference
        try:
            input_values = input_values.cuda()
            print(f"[DEBUG] Input values transferred to CUDA. Device: {input_values.device}")
            model = model.cuda()
            print(f"[DEBUG] Model transferred to CUDA")
        except Exception as e:
            print(f"[DEBUG] Error transferring to CUDA: {e}")
            raise

        if attention_mask is not None:
            print(f"[DEBUG] Using attention_mask with shape: {attention_mask.shape}")
            attention_mask = attention_mask.cuda()
            inputs = (input_values, attention_mask)
        else:
            print("[DEBUG] No attention_mask provided")
            inputs = (input_values,)

        # Create CUDA events for timing GPU operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # INFERENCE!
        print("[DEBUG] Running model inference on CUDA")
        start.record()
        try:
            preds = model(*inputs)
            print(f"[DEBUG] CUDA inference successful. Output type: {type(preds)}")
            if isinstance(preds, dict):
                print(f"[DEBUG] Prediction keys: {preds.keys()}")
                for k, v in preds.items():
                    if torch.is_tensor(v):
                        print(f"[DEBUG] Prediction [{k}] shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
                    else:
                        print(f"[DEBUG] Prediction [{k}] type: {type(v)}")
            elif torch.is_tensor(preds):
                print(f"[DEBUG] Prediction tensor shape: {preds.shape}, dtype: {preds.dtype}, device: {preds.device}")
        except Exception as e:
            print(f"[DEBUG] Error during CUDA inference: {e}")
            raise
        end.record()

        # Synchronize to ensure all GPU operations are finished
        torch.cuda.synchronize()

        # Calculate latency between start and end events
        latency = start.elapsed_time(end)
        print(f"[DEBUG] CUDA inference latency: {latency:.2f} ms")

        if isinstance(preds, dict):
            detached_dict = {}
            for k, v in preds.items():
                if torch.is_tensor(v):
                    detached_dict[k] = v.detach().cpu()
                    print(f"[DEBUG] Transferred {k} to CPU")
                else:
                    detached_dict[k] = v
            preds = detached_dict
        elif torch.is_tensor(preds):
            preds = preds.detach().cpu()
            print(f"[DEBUG] Transferred tensor output to CPU")

        return preds, latency

    def infer_trt_cuda(self, trt_context, input_values, attention_mask=None):
        print(f"[DEBUG] Starting TensorRT inference with input shape: {input_values.shape}")
        input_values_np = input_values.cpu().numpy()  # ensure on CPU before converting
        if attention_mask is not None:
            print(f"[DEBUG] Using attention_mask with shape: {attention_mask.shape}")
            attention_mask_np = attention_mask.cpu().numpy()
            input_arrays = [input_values_np, attention_mask_np]
        else:
            print("[DEBUG] No attention_mask provided")
            input_arrays = [input_values_np]
            
        bufferH = []
        for i, arr in enumerate(input_arrays):
            bufferH.append(np.ascontiguousarray(arr))
            print(f"[DEBUG] Input {i} shape: {arr.shape}, dtype: {arr.dtype}")

        for i in range(self.n_Input, self.num_io):
            output_shape = self.context.get_tensor_shape(self.lTensorName[i])
            output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i]))
            print(f"[DEBUG] Creating output buffer {i} with shape: {output_shape}, dtype: {output_dtype}")
            bufferH.append(
                np.empty(
                    output_shape,
                    dtype=output_dtype,
                )
            )
        
        print("[DEBUG] Allocating device memory")
        bufferD = []
        for i in range(self.num_io):
            cuda_result = cudart.cudaMalloc(bufferH[i].nbytes)
            if cuda_result[0] != 0:  # 0 is cudaSuccess
                print(f"[DEBUG] CUDA malloc failed with code {cuda_result[0]}")
            bufferD.append(cuda_result[1])
            print(f"[DEBUG] Allocated {bufferH[i].nbytes} bytes for tensor {i}")

        print("[DEBUG] Copying input data to device")
        for i in range(self.n_Input):
            cuda_result = cudart.cudaMemcpy(
                bufferD[i],
                bufferH[i].ctypes.data,
                bufferH[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
            if cuda_result != 0:  # 0 is cudaSuccess
                print(f"[DEBUG] CUDA memcpy H2D failed with code {cuda_result}")

        print("[DEBUG] Setting tensor addresses")
        for i in range(self.num_io):
            self.context.set_tensor_address(self.lTensorName[i], int(bufferD[i]))
            print(f"[DEBUG] Set address for tensor {self.lTensorName[i]}")

        # Create CUDA events for timing GPU operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # INFERENCE!
        print("[DEBUG] Running TensorRT inference")
        start.record()
        try:
            result = self.context.execute_async_v3(0)
            if not result:
                print("[DEBUG] TensorRT execution failed")
            else:
                print("[DEBUG] TensorRT execution succeeded")
        except Exception as e:
            print(f"[DEBUG] Error during TensorRT execution: {e}")
            raise
        end.record()

        # Synchronize to ensure all GPU operations are finished
        torch.cuda.synchronize()

        # Calculate latency between start and end events
        latency = start.elapsed_time(end)
        print(f"[DEBUG] TensorRT inference latency: {latency:.2f} ms")

        # Copying data from device to host and collecting output tensors
        print("[DEBUG] Copying output data from device to host")
        output_data = []
        for i in range(self.n_Input, self.num_io):
            cuda_result = cudart.cudaMemcpy(
                bufferH[i].ctypes.data,
                bufferD[i],
                bufferH[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            )
            if cuda_result != 0:  # 0 is cudaSuccess
                print(f"[DEBUG] CUDA memcpy D2H failed with code {cuda_result}")
            output_data.append(bufferH[i])
            print(f"[DEBUG] Copied output {i} with shape: {bufferH[i].shape}, dtype: {bufferH[i].dtype}")

        # Flatten output if it consists of only one item
        output_data = output_data[0] if len(output_data) == 1 else output_data
        print(f"[DEBUG] Final output data type: {type(output_data)}")

        print("[DEBUG] Freeing CUDA memory")
        for b in bufferD:
            cudart.cudaFree(b)

        if len(output_data) == 1:
            output_data = output_data[0]
        # Convert the raw scores from numpy array to PyTorch tensor
        preds_tensor = torch.tensor(output_data, device="cpu", dtype=torch.float32)
        print(f"[DEBUG] Converted output to tensor with shape: {preds_tensor.shape}")

        return preds_tensor, latency

    def infer_onnx_cpu(self, ort_inference_session, input_values, attention_mask=None):
        print(f"[DEBUG] Starting ONNX CPU inference with input shape: {input_values.shape}")
        # Convert PyTorch tensor to numpy array for ONNX Runtime
        input_values_np = input_values.cpu().numpy()
        if attention_mask is not None:
            print(f"[DEBUG] Using attention_mask with shape: {attention_mask.shape}")
            attention_mask_np = attention_mask.cpu().numpy()
            inputs = {"input": {"input_values": input_values_np, "attention_mask": attention_mask_np}}
        else:
            print("[DEBUG] No attention_mask provided")
            inputs = {"input": {"input_values": input_values_np}}

        print(f"[DEBUG] ONNX input structure: {list(inputs.keys())}")
        print(f"[DEBUG] ONNX input[input] keys: {list(inputs['input'].keys())}")

        # Run inference using ONNX Runtime
        start_time = time.time()
        try:
            print("[DEBUG] Running ONNX inference")
            output_data = ort_inference_session.run(None, inputs)
            print(f"[DEBUG] ONNX inference successful. Output length: {len(output_data)}")
            for i, out in enumerate(output_data):
                print(f"[DEBUG] Output {i} shape: {out.shape}, dtype: {out.dtype}")
        except Exception as e:
            print(f"[DEBUG] Error during ONNX inference: {e}")
            raise

        # End timing CPU operations
        end_time = time.time()

        # Calculate latency in milliseconds
        latency = (
            end_time - start_time
        ) * 1000.0  # Convert from seconds to milliseconds
        print(f"[DEBUG] ONNX CPU inference latency: {latency:.2f} ms")

        # Flatten output if it consists of only one item
        output_data = output_data[0] if len(output_data) == 1 else output_data
        print(f"[DEBUG] Final output data type: {type(output_data)}")

        # Convert the raw scores from numpy array back to PyTorch tensor
        preds_tensor = torch.from_numpy(
            output_data
        ).float()  # Ensure tensor is on CPU and in float32 format
        print(f"[DEBUG] Converted output to tensor with shape: {preds_tensor.shape}")

        return preds_tensor, latency

    def infer_onnx_cuda(self, ort_inference_session, input_values, attention_mask=None):
        print(f"[DEBUG] Starting ONNX CUDA inference with input shape: {input_values.shape}")
        input_values_np = input_values.cpu().numpy()

        if attention_mask is not None:
            print(f"[DEBUG] Using attention_mask with shape: {attention_mask.shape}")
            attention_mask_np = attention_mask.cpu().numpy()
            inputs = {"input": {"input_values": input_values_np, "attention_mask": attention_mask_np}}
        else:
            print("[DEBUG] No attention_mask provided")
            inputs = {"input": {"input_values": input_values_np}}
        
        print(f"[DEBUG] ONNX input structure: {list(inputs.keys())}")
        
        # Create CUDA events for timing GPU operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        try:
            print("[DEBUG] Running ONNX inference on CUDA")
            output_data = ort_inference_session.run(None, inputs)
            print(f"[DEBUG] ONNX CUDA inference successful. Output length: {len(output_data)}")
            for i, out in enumerate(output_data):
                print(f"[DEBUG] Output {i} shape: {out.shape}, dtype: {out.dtype}")
        except Exception as e:
            print(f"[DEBUG] Error during ONNX CUDA inference: {e}")
            raise
        end.record()

        # Synchronize to ensure all GPU operations are finished
        torch.cuda.synchronize()

        # Calculate latency between start and end events
        latency = start.elapsed_time(end)
        print(f"[DEBUG] ONNX CUDA inference latency: {latency:.2f} ms")

        # Flatten output if it consists of only one item
        output_data = output_data[0] if len(output_data) == 1 else output_data
        print(f"[DEBUG] Final output data type: {type(output_data)}")

        # Convert the raw scores from numpy array to PyTorch tensor
        preds_tensor = torch.tensor(output_data, device="cpu", dtype=torch.float32)
        print(f"[DEBUG] Converted output to tensor with shape: {preds_tensor.shape}")

        return preds_tensor, latency

    def evaluate(self):
        print(f"[DEBUG] Starting evaluation on {self.model_name}")
        self.logger.info(f"Starting transformation analysis on {self.model_name}")

        num_GPU_warmup_batches = self.config["num_GPU_warmup_batches"]
        task = self.config["task"]
        print(f"[DEBUG] Task: {task}, Warmup batches: {num_GPU_warmup_batches}")

        # Check if the model requires attention_mask
        if hasattr(self.model, 'forward') and callable(self.model.forward):
            print(f"[DEBUG] Model forward method signature: {inspect.signature(self.model.forward)}")
            
        requires_attention_mask = self.config.get(
            "requires_attention_mask",
            hasattr(self.model, 'forward') and callable(self.model.forward) and "attention_mask" in inspect.signature(self.model.forward).parameters
        )
        print(f"[DEBUG] Model requires attention mask: {requires_attention_mask}")

        # ---------- 1) SET UP METRICS BASED ON TASK ----------
        print("[DEBUG] Setting up metrics for task: " + task)
        if task == "cls":
            print(f"[DEBUG] Creating classification metrics with {self.num_of_classes} classes")
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
            print("[DEBUG] Creating CTC metrics")
            wer_metric = jiwer.wer
            decoder = self.config.get("decoder", None)
            print(f"[DEBUG] Decoder provided: {decoder is not None}")
            
            beam_width = self.config.get("beam_width", 10)
            print(f"[DEBUG] Beam width: {beam_width}")
            
            tokenizer = self.config.get("tokenizer")
            print(f"[DEBUG] Tokenizer provided: {tokenizer is not None}")
            
            ctc_head = self.config.get("ctc_head", None)
            print(f"[DEBUG] CTC head provided: {ctc_head is not None}")
            if ctc_head is not None:
                print(f"[DEBUG] CTC head type: {type(ctc_head)}")
                
            padding_value = self.config.get("padding_value", -100)
            print(f"[DEBUG] Padding value: {padding_value}")

            # We'll collect WER from each batch
            batch_wers = []

        else:
            print(f"[DEBUG] Unsupported task type: {task}")
            raise Exception(
                f"Unsupported task type {task}. Please set a supported task type in the config file."
            )
    
        # ---------- 2) PREPARE DATA LOADER (TEST OR VALIDATION) ----------
        if "test" in self.config and self.config["test"]:
            print("[DEBUG] Using test dataloader")
            dataloader = self.config["data_module"].test_dataloader()
            dataset = "Test"
        else:
            print("[DEBUG] Using validation dataloader")
            dataloader = self.config["data_module"].val_dataloader()
            dataset = "Validation"

        print(f"[DEBUG] {dataset} dataloader created")

        # ---------- 3) ARRAYS FOR LATENCIES & POWER FOR ALL TASKS ----------
        latencies = []
        gpu_power_usages = []
        rtfs = [] 

        # ---------- 4) MAIN EVALUATION LOOP ----------
        from transformers.feature_extraction_utils import BatchFeature
        print("[DEBUG] Starting main evaluation loop")

        # Inside the evaluate() method of RuntimeAnalysis, in the main evaluation loop:
        for j, batch in enumerate(dataloader):
            print(f"[DEBUG] Processing batch {j+1}")
            print(f"[DEBUG] Batch type: {type(batch)}")
            
            # Convert BatchFeature to dict if necessary.
            if isinstance(batch, BatchFeature):
                print("[DEBUG] Batch is a BatchFeature, converting to dict")
                batch = batch.data
                
            if isinstance(batch, dict):
                print(f"[DEBUG] Batch is a dict with keys: {batch.keys()}")
                xs = batch["input_values"]
                ys = batch["labels"]
                attention_mask = batch.get("attention_mask", None)
                print(f"[DEBUG] Extracted input_values with shape: {xs.shape}")
                print(f"[DEBUG] Extracted labels with shape: {ys.shape}")
                if attention_mask is not None:
                    print(f"[DEBUG] Extracted attention_mask with shape: {attention_mask.shape}")
            elif isinstance(batch, (list, tuple)):
                print(f"[DEBUG] Batch is a {type(batch).__name__} with length: {len(batch)}")
                xs = batch[0]
                ys = batch[1]
                attention_mask = batch[2] if len(batch) > 2 else None
                print(f"[DEBUG] Extracted inputs with shape: {xs.shape}")
                print(f"[DEBUG] Extracted labels with shape: {ys.shape}")
                if attention_mask is not None:
                    print(f"[DEBUG] Extracted attention_mask with shape: {attention_mask.shape}")
            else:
                error_msg = f"Unsupported batch format: {type(batch)}"
                self.logger.error(error_msg)
                print(f"[DEBUG] {error_msg}")
                raise TypeError(f"Expected batch to be dict or list/tuple, got {type(batch)}")
            
            # Continue with your processing...
            
            # Stop if we've exceeded our number of evaluation batches, or if the batch is incomplete
            if j >= self.config["num_batches"] or xs.shape[0] != self.config["batch_size"]:
                print(f"[DEBUG] Batch {j+1} - xs shape: {xs.shape}, Expected batch size: {self.config['batch_size']}")
                print(f"[DEBUG] Breaking loop: {'max batches reached' if j >= self.config['num_batches'] else 'incomplete batch'}")
                break

            # Power monitoring (start)
            print("[DEBUG] Starting power monitoring")
            power_monitor = PowerMonitor(self.config)
            try:
                power_monitor.start()
                print("[DEBUG] Power monitoring started successfully")
            except Exception as e:
                print(f"[DEBUG] Error starting power monitor: {e}")

            # Clear GPU caches & sync
            print("[DEBUG] Synchronizing CUDA and clearing cache")
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[DEBUG] Error during CUDA sync/empty: {e}")

            # Determine model inputs based on attention mask requirement
            if requires_attention_mask:
                print("[DEBUG] Using model input with attention mask")
                if attention_mask is None:
                    print("[DEBUG] Warning: Model requires attention mask but none provided")
                    # Create a default attention mask if needed
                    attention_mask = torch.ones_like(xs, dtype=torch.long)
                    print(f"[DEBUG] Created default attention mask with shape: {attention_mask.shape}")
                attention_mask = attention_mask.to(self.config["accelerator"])
                model_input = (xs, attention_mask)
            else:
                print("[DEBUG] Using model input without attention mask")
                model_input = (xs,)

            # ---------------- (A) RUN INFERENCE (TRT, ONNX, or MaseGraph) ----------------
            print(f"[DEBUG] Running inference with model type: {type(self.model)}")
            if isinstance(self.model, trt.IExecutionContext):
                # TensorRT
                print("[DEBUG] Running TensorRT inference")
                if self.config["accelerator"] != "cuda":
                    error_msg = "TensorRT inference is only supported on CUDA devices."
                    print(f"[DEBUG] Error: {error_msg}")
                    raise Exception(error_msg)
                try:
                    preds, latency = self.infer_trt_cuda(self.model, *model_input)
                    print(f"[DEBUG] TensorRT inference completed with latency: {latency:.2f} ms")
                except Exception as e:
                    print(f"[DEBUG] TensorRT inference failed: {e}")
                    raise

            elif isinstance(self.model, ort.InferenceSession):
                # ONNX Runtime
                print(f"[DEBUG] Running ONNX inference on {self.config['accelerator']}")
                if self.config["accelerator"] == "cpu":
                    try:
                        preds, latency = self.infer_onnx_cpu(self.model, *model_input)
                        print(f"[DEBUG] ONNX CPU inference completed with latency: {latency:.2f} ms")
                    except Exception as e:
                        print(f"[DEBUG] ONNX CPU inference failed: {e}")
                        raise
                elif self.config["accelerator"] == "cuda":
                    try:
                        preds, latency = self.infer_onnx_cuda(self.model, *model_input)
                        print(f"[DEBUG] ONNX CUDA inference completed with latency: {latency:.2f} ms")
                    except Exception as e:
                        print(f"[DEBUG] ONNX CUDA inference failed: {e}")
                        raise
                else:
                    error_msg = f"ONNX inference is not support by device {self.config['accelerator']}."
                    print(f"[DEBUG] Error: {error_msg}")
                    raise Exception(error_msg)

            else:
                # MaseGraph or raw PyTorch
                print(f"[DEBUG] Running MaseGraph inference on {self.config['accelerator']}")
                if self.config["accelerator"] == "cpu":
                    try:
                        preds, latency = self.infer_mg_cpu(self.model, *model_input)
                        print(f"[DEBUG] MaseGraph CPU inference completed with latency: {latency:.2f} ms")
                    except Exception as e:
                        print(f"[DEBUG] MaseGraph CPU inference failed: {e}")
                        raise
                elif self.config["accelerator"] == "cuda":
                    try:
                        print(f"[DEBUG] Running MaseGraph inference on batch {j+1}")
                        preds, latency = self.infer_mg_cuda(self.model, *model_input)
                        print(f"[DEBUG] MaseGraph CUDA inference completed with latency: {latency:.2f} ms")
                    except Exception as e:
                        print(f"[DEBUG] MaseGraph CUDA inference failed: {e}")
                        raise
                else:
                    error_msg = f"MaseGraph inference is not support by device {self.config['accelerator']}."
                    print(f"[DEBUG] Error: {error_msg}")
                    raise Exception(error_msg)

            # Power monitoring (stop)
            print("[DEBUG] Stopping power monitoring")
            try:
                power_monitor.stop()
                power_monitor.join()
                print("[DEBUG] Power monitoring stopped successfully")
                print(f"[DEBUG] Power readings: {power_monitor.power_readings}")
            except Exception as e:
                print(f"[DEBUG] Error stopping power monitor: {e}")

            # Skip warmup batches
            if j < num_GPU_warmup_batches:
                print(f"[DEBUG] Skipping warmup batch {j+1}")
                continue

            # Record latency
            latencies.append(latency)
            print(f"[DEBUG] Recorded latency: {latency:.2f} ms, Average so far: {sum(latencies)/len(latencies):.2f} ms")

            # Compute average power usage for this batch
            avg_power = (
                sum(power_monitor.power_readings) / len(power_monitor.power_readings)
                if power_monitor.power_readings
                else 0
            )
            gpu_power_usages.append(avg_power)
            print(f"[DEBUG] Recorded power usage: {avg_power:.2f} W, Average so far: {sum(gpu_power_usages)/len(gpu_power_usages):.2f} W")

            # ---------- (B) METRICS DEPENDING ON TASK ----------
            if task == "cls":
                # Classification logic
                print("[DEBUG] Computing classification metrics")
                # preds: [batch_size, num_classes]
                # ys: [batch_size]

                try:
                    # Cross-entropy (classification) loss
                    loss = torch.nn.functional.cross_entropy(preds, ys)
                    losses.append(loss.item())
                    print(f"[DEBUG] Classification loss: {loss.item():.4f}")

                    # Accuracy (MulticlassAccuracy)
                    acc = metric(preds, ys)
                    accs.append(acc.item())
                    print(f"[DEBUG] Classification accuracy: {acc.item():.4f}")

                    # Update precision, recall, F1
                    preds_labels = torch.argmax(preds, dim=1)
                    precision_metric(preds_labels, ys)
                    recall_metric(preds_labels, ys)
                    f1_metric(preds_labels, ys)
                    print("[DEBUG] Updated precision, recall, and F1 metrics")
                except Exception as e:
                    print(f"[DEBUG] Error computing classification metrics: {e}")
                    raise

            elif task == "ctc":
                print("[DEBUG] Computing CTC metrics")
                # Real-Time Factor (RTF) = Latency / (1 / sample_rate)
                try:
                    max_length = xs.shape[1]
                    sample_rate = self.config.get("sample_rate", 16000)
                    audio_duration = max_length / sample_rate
                    rtf = (latency / 1000.0) / audio_duration
                    rtfs.append(rtf)
                    print(f"[DEBUG] Audio duration: {audio_duration:.4f}s, RTF: {rtf:.4f}")
                except Exception as e:
                    print(f"[DEBUG] Error computing RTF: {e}")
                    raise

                # WER CTC logic
                # preds: [batch_size, time_steps, vocab_size] or [batch_size, *]
                # ys: [batch_size, time_steps]

                # Apply CTC decoding to get predicted text
                if ctc_head is None:
                    error_msg = "CTC head must be provided in config for full model evaluation"
                    print(f"[DEBUG] Error: {error_msg}")
                    raise Exception(error_msg)
                
                try:
                    print("[DEBUG] Processing CTC head output")
                    ctc_head = ctc_head.cpu()
                    
                    if isinstance(preds, dict) and "last_hidden_state" in preds:
                        print(f"[DEBUG] Found last_hidden_state in predictions with shape: {preds['last_hidden_state'].shape}")
                        encoder_output = preds["last_hidden_state"].cpu()
                    else:
                        print(f"[DEBUG] Using raw predictions as encoder output, type: {type(preds)}")
                        if torch.is_tensor(preds):
                            print(f"[DEBUG] Prediction tensor shape: {preds.shape}")
                        encoder_output = preds.cpu() if torch.is_tensor(preds) else preds

                    print("[DEBUG] Applying CTC head to encoder output")
                    predictions = ctc_head(encoder_output)
                    print(f"[DEBUG] CTC head output type: {type(predictions)}")

                    if torch.is_tensor(predictions):
                        print(f"[DEBUG] CTC predictions tensor shape: {predictions.shape}")
                        predictions = predictions.detach().cpu()
                    else:
                        print(f"[DEBUG] CTC predictions non-tensor type: {type(predictions)}")
                        predictions = torch.tensor(predictions).cpu()

                    preds_np = predictions.numpy()
                    print(f"[DEBUG] Predictions numpy array shape: {preds_np.shape}")

                    pred_texts = []
                    label_texts = []

                    print("[DEBUG] Decoding predictions")
                    for i in range(preds_np.shape[0]):
                        sample_logits = torch.from_numpy(preds_np[i])
                        sample_log_probs = sample_logits.log_softmax(dim=-1).cpu().numpy()
                        if decoder is not None:
                            print(f"[DEBUG] Decoding sample {i+1} with beam width {beam_width}")
                            transcription = decoder.decode(sample_log_probs, beam_width=beam_width)
                            print(f"[DEBUG] Transcription for sample {i+1}: '{transcription}'")
                        else:
                            error_msg = "Decoder must be provided for CTC runtime analysis. Pass 'decoder' in config."
                            print(f"[DEBUG] Error: {error_msg}")
                            raise Exception(error_msg)
                        pred_texts.append(transcription.lower())

                    print("[DEBUG] Decoding ground truth labels")
                    for label_seq in ys:
                        print(f"[DEBUG] Label sequence shape: {label_seq.shape}, min: {label_seq.min()}, max: {label_seq.max()}")
                        label_filtered = [token for token in label_seq if token != padding_value]
                        print(f"[DEBUG] Filtered {len(label_seq) - len(label_filtered)} padding tokens")
                        try:
                            label_text = tokenizer.decode(label_filtered, skip_special_tokens=True)
                            print(f"[DEBUG] Decoded label: '{label_text}'")
                        except Exception as e:
                            print(f"[DEBUG] Error decoding label: {e}")
                            label_text = ""
                        label_texts.append(label_text.lower())

                    # Now compute batch WER
                    print(f"[DEBUG] Computing WER for batch {j+1}")
                    for i, (pred, label) in enumerate(zip(pred_texts, label_texts)):
                        print(f"[DEBUG] Sample {i+1}:")
                        print(f"[DEBUG]   Prediction: '{pred}'")
                        print(f"[DEBUG]   Reference:  '{label}'")
                    
                    batch_wer = wer_metric(pred_texts, label_texts)
                    wer_value = batch_wer.item() if torch.is_tensor(batch_wer) else batch_wer
                    batch_wers.append(wer_value)
                    print(f"[DEBUG] Batch WER: {wer_value:.4f}")
                except Exception as e:
                    print(f"[DEBUG] Error in CTC processing: {e}")
                    import traceback
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                    # Don't raise to allow collection of other metrics

        # ---------- 5) AFTER LOOP, COMPUTE FINAL METRICS BASED ON TASK ----------
        print("[DEBUG] Computing final metrics")
        print(f"[DEBUG] Collected {len(latencies)} latency measurements")
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        print(f"[DEBUG] Average latency: {avg_latency:.2f} ms")
        
        print(f"[DEBUG] Collected {len(gpu_power_usages)} power measurements")
        avg_gpu_power_usage = sum(gpu_power_usages) / len(gpu_power_usages) if gpu_power_usages else 0
        print(f"[DEBUG] Average GPU power usage: {avg_gpu_power_usage:.2f} W")
        
        # Energy in mWh = Power (W) * Time (h) * 1000 (m)
        avg_gpu_energy_usage = (avg_gpu_power_usage * 1000) * (avg_latency / 3600000)
        print(f"[DEBUG] Average GPU energy usage: {avg_gpu_energy_usage:.5f} mWh")

        if task == "cls":
            # Classification final metrics
            print("[DEBUG] Computing final classification metrics")
            print(f"[DEBUG] Collected {len(accs)} accuracy measurements")
            avg_acc = sum(accs) / len(accs) if accs else 0
            print(f"[DEBUG] Average accuracy: {avg_acc:.4f}")
            
            print(f"[DEBUG] Collected {len(losses)} loss measurements")
            avg_loss = sum(losses) / len(losses) if losses else 0
            print(f"[DEBUG] Average loss: {avg_loss:.4f}")

            avg_precision = precision_metric.compute()
            avg_recall = recall_metric.compute()
            avg_f1 = f1_metric.compute()

            # Convert to float
            avg_precision = avg_precision.item() if torch.is_tensor(avg_precision) else avg_precision
            avg_recall = avg_recall.item() if torch.is_tensor(avg_recall) else avg_recall
            avg_f1 = avg_f1.item() if torch.is_tensor(avg_f1) else avg_f1
            
            print(f"[DEBUG] Precision: {avg_precision:.4f}")
            print(f"[DEBUG] Recall: {avg_recall:.4f}")
            print(f"[DEBUG] F1: {avg_f1:.4f}")

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
            print("[DEBUG] Computing final CTC metrics")
            print(f"[DEBUG] Collected {len(rtfs)} RTF measurements")
            avg_rtf = sum(rtfs) / len(rtfs) if rtfs else 0
            print(f"[DEBUG] Average RTF: {avg_rtf:.4f}")

            # CTC final metrics
            print(f"[DEBUG] Collected {len(batch_wers)} WER measurements")
            avg_wer = sum(batch_wers) / len(batch_wers) if batch_wers else 0
            print(f"[DEBUG] Average WER: {avg_wer:.4f}")

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
        print(f"[DEBUG] Final results for {self.model_name}:")
        print(formatted_metrics)

        return results