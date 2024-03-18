import logging
import torch
import json
from pathlib import PosixPath
import tensorrt as trt
from chop.ir import MaseGraph
from .utils import PowerMonitor, prepare_save_path
import sys
import logging 
import os
from tabulate import tabulate
import torch
import torchmetrics
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
import onnxruntime as ort
from cuda import cudart

dtype_map = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
}

def tensorrt_analysis_pass(model, pass_args=None):
    analysis = QuantizationAnalysis(model, pass_args)
    results = analysis.evaluate()

    results_path = prepare_save_path(method='analysis', suffix='json')
    # with open(results_path, 'w') as json_file:
    #     json.dump(results, json_file, indent=4)
    
    return model, results

class QuantizationAnalysis():
    def __init__(self, model, config):

        # Istantiate default performance analyzer args
        if 'num_batches' not in config.keys():
            config['num_batches'] = 500
            config['num_GPU_warmup_batches'] = 5
            config['test'] = True
        
        self.config = config

        self.logger = logging.getLogger(__name__)
        self.num_of_classes = self.config['data_module'].dataset_info.num_classes

        match model:
            case MaseGraph():
                # Check if model is mase graph
                self.model = model.model
                self.model_name = self.config['model']
            
            case PosixPath() as path:
                match path.suffix:
                    case '.trt':
                        # Load the serialized TensorRT engine
                        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                        with open(path, "rb") as f:
                            self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
                        self.runtime = trt.Runtime(TRT_LOGGER) 
                        self.num_io = self.engine.num_io_tensors
                        self.lTensorName = [self.engine.get_tensor_name(i) for i in range(self.num_io)]
                        self.n_Input = [self.engine.get_tensor_mode(self.lTensorName[i]) for i in range(self.num_io)].count(trt.TensorIOMode.INPUT)
                        self.context = self.engine.create_execution_context()
                        self.model = self.context
                        self.summarize()
                        self.model_name = f"{self.config['model']}-quantized"
                        
                    case '.onnx':
                        # Load the exported ONNX model into an ONNXRuntime inference session
                        self.model = ort.InferenceSession(path, providers=[config['execution_provider']])
                        self.model_name = f"{self.config['model']}-onnx"
                    case _:
                        # If file type is neither .trt nor .onnx
                        raise Exception("Model must be a MaseGraph or a path to a trt file. Have you run the quantization pass?")
            case _:
                # If model is neither MaseGraph nor PosixPath
                raise Exception("Model must be a MaseGraph or a PosixPath to a trt file. Have you run the quantization pass?")

    def summarize(self):
        io_info_lines = [
            "Index | Type    | DataType | Static Shape         | Dynamic Shape        | Name",
            "------|---------|----------|----------------------|----------------------|-----------------------"
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
            dtype_str = str(dtype).split('.')[-1]  # Convert from DataType.FLOAT to FLOAT

            # Format the IO information
            io_info = f"{binding_index:<5} | {io_type:<7} | {dtype_str:<8} | {str(static_shape):<22} | {str(dynamic_shape):<22} | {name}"
            io_info_lines.append(io_info)

        # Join all IO information into a single string and log it
        io_info_str = "\n".join(io_info_lines)
        self.logger.info(f"\nTensorRT Engine Input/Output Information:\n{io_info_str}")

    def infer_mg(self, model, input_data):
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

    def infer_trt(self, trt_context, input_data):
        bufferH = []
        bufferH.append(np.ascontiguousarray(input_data))
        for i in range(self.n_Input, self.num_io):
            bufferH.append(np.empty(self.context.get_tensor_shape(self.lTensorName[i]), dtype=trt.nptype(self.engine.get_tensor_dtype(self.lTensorName[i]))))
        bufferD = []
        for i in range(self.num_io):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(self.n_Input):
            cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

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
            bufferH[i] for i in range(self.n_Input, self.num_io)
            for _ in [cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)]
        ]

        # Flatten output if it consists of only one item
        output_data = output_data[0] if len(output_data) == 1 else output_data

        for b in bufferD:
            cudart.cudaFree(b)

        # Convert the raw scores from numpy array to PyTorch tensor
        preds_tensor = torch.tensor(output_data, device='cpu', dtype=torch.float32)

        return preds_tensor, latency
    
    def infer_onnx(self, ort_inference_session, input_data):
        # Create CUDA events for timing GPU operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output_data = ort_inference_session.run(None, {'input': input_data.numpy()})
        end.record()
        
        # Synchronize to ensure all GPU operations are finished
        torch.cuda.synchronize()

        # Calculate latency between start and end events
        latency = start.elapsed_time(end)

        # Flatten output if it consists of only one item
        output_data = output_data[0] if len(output_data) == 1 else output_data

        # Convert the raw scores from numpy array to PyTorch tensor
        preds_tensor = torch.tensor(output_data, device='cpu', dtype=torch.float32)

        return preds_tensor, latency
    
    def evaluate(self):
        self.logger.info("Starting transformation analysis")

        num_GPU_warmup_batches = self.config['num_GPU_warmup_batches']

        # Instantiate metrics with the specified task type
        if self.config['task'] == 'cls':
            metric = torchmetrics.classification.MulticlassAccuracy(num_classes=self.num_of_classes)
            precision_metric = torchmetrics.Precision(num_classes=self.num_of_classes, average='weighted', task='multiclass')
            recall_metric = torchmetrics.Recall(num_classes=self.num_of_classes, average='weighted', task='multiclass')
            f1_metric = torchmetrics.F1Score(num_classes=self.num_of_classes, average='weighted', task='multiclass')            

        # Initialize lists to store metrics for each configuration
        recorded_accs = []
        latencies = []
        gpu_power_usages = []
        accs, losses = [], []

        if 'test' in self.config and self.config['test']:
            dataloader = self.config['data_module'].test_dataloader()
            dataset = 'Test'
        else:
            dataloader = self.config['data_module'].val_dataloader()
            dataset = 'Validation'
        
        # Iterate over batches in the validation/train dataset
        for j, (xs, ys) in enumerate(dataloader):
            # Break the loop after processing the specified number of batches
            if j >= self.config['num_batches']:
                break

            # Instantiate and start the power monitor
            power_monitor = PowerMonitor(self.config)
            power_monitor.start()

            # Synchronize to ensure all GPU operations are finished
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            if isinstance(self.model, trt.IExecutionContext):
                preds, latency = self.infer_trt(self.model, xs)
                
            elif isinstance(self.model, ort.InferenceSession):

                # Since dynamic batching is not supported, the last test batch is not dropped, the last batch 
                # size may mismatch with the expected size. Skip the sample if a mismatch if detected.
                if xs.shape[0] != self.config['batch_size']:
                    power_monitor.stop()
                    power_monitor.join() 

                    print(f"Dynamic batching not supported: received batch of size {xs.shape[0]} but expected size {self.config['batch_size']}.")
                    continue

                preds, latency = self.infer_onnx(self.model, xs)

            else:
                preds, latency = self.infer_mg(self.model, xs)  # Run model prediction
            
            # Stop the power monitor and calculate average power
            power_monitor.stop()
            power_monitor.join()  # Ensure monitoring thread has finished

            # Skip the first number of iterations to allow the model to warm up
            if j < num_GPU_warmup_batches:
                continue

            latencies.append(latency)

            avg_power = sum(power_monitor.power_readings) / len(power_monitor.power_readings) if power_monitor.power_readings else 0
            # Store the calculated average power
            gpu_power_usages.append(avg_power)
            
            # Calculate accuracy and loss for the batch
            loss = torch.nn.functional.cross_entropy(preds, ys)
            acc = metric(preds, ys)
            accs.append(acc)
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
        avg_gpu_energy_usage = (avg_gpu_power_usage* 1000) * (avg_latency / 3600000)
        
        metrics = [
            ["Average " + dataset + " Accuracy", f"{acc_avg:.5g}"],
            ["Average Precision", f"{avg_precision:.5g}"],
            ["Average Recall", f"{avg_recall:.5g}"],
            ["Average F1 Score", f"{avg_f1:.5g}"],
            ["Average Loss", f"{loss_avg:.5g}"],
            ["Average Latency", f"{avg_latency:.5g} ms"],
            ["Average GPU Power Usage", f"{avg_gpu_power_usage:.5g} W"],
            ["Inference Energy Consumption", f"{avg_gpu_energy_usage:.5g} mWh"]
        ]

        # Formatting the table with tabulate
        formatted_metrics = tabulate(metrics, headers=['Metric', 'Value'], tablefmt="pretty", floatfmt=".5g")

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
            "Inference Energy Consumption": avg_gpu_energy_usage
        }
        return results