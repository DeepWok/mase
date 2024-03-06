from copy import copy, deepcopy
import logging
import torch
import os
import sys
from pathlib import Path
from datetime import datetime
import tensorrt as trt
from pathlib import Path
from datetime import datetime

from pytorch_quantization import quant_modules, calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn import TensorQuantizer
from pytorch_quantization.tensor_quant import QuantDescriptor

from chop.passes.graph.utils import get_mase_op, get_mase_type, get_node_actual_target
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass
from ....utils import deepcopy_mase_graph
import sys
import logging 
import os
from pathlib import Path
from pprint import pprint as pp
import time
import pynvml
import threading
import time
import torch
from torchmetrics.classification import MulticlassAccuracy
import torchmetrics
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This is needed for initializing CUDA driver
# from chop.passes.graph.analysis.quantization import calculate_flops_pass #TODO add FLOPS
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)


def tensorrt_analysis_pass(original_graph, trt_engine_path, pass_args=None):
    analysis = QuantizationAnalysis(original_graph, trt_engine_path, pass_args)
    analysis.evaluate()

class QuantizationAnalysis():
    def __init__(self, original_graph, trt_engine_path, config):
        self.original_graph = original_graph

        # Load the serialized TensorRT engine
        with open(trt_engine_path, "rb") as f:
            engine_data = f.read()
        self.trt_engine = trt_runtime.deserialize_cuda_engine(engine_data)

        # Allocate buffers for input and output
        self.context = self.engine.create_execution_context()
        # NOTE: You should calculate or know your input/output sizes.
        #       They depend on your model's configuration.
        self.input_nbytes = ...  
        self.output_nbytes = ...
        self.input_memory = cuda.mem_alloc(self.input_nbytes)
        self.output_memory = cuda.mem_alloc(self.output_nbytes)

        # Create a stream for CUDA operations
        self.stream = cuda.Stream()

        self.config = config
        self.logger = logging.getLogger(__name__)
        self.power_monitor = Power_Monitor(self.config)
    
    def evaluate(self):
        self.logger.info("Starting TensorRT transformation analysis")
        for analysis in self.config["analysis"]:
            match analysis:
                case "latency":
                    self.eval()
                case "accuracy":
                    self.eval()
                case _:
                    raise ValueError(f"Unsupported analysis: {analysis}")
        self.logger.info("Finished TensorRT transformation analysis")

    def infer_mg(self, graph, input_data):
        graph.model(input_data)

    def infer_trt(self, trt_context, input_data):
        # Copy input data to the GPU
        cuda.memcpy_htod_async(self.input_memory, input_data, self.stream)
        # Execute the model
        trt_context.execute_async(batch_size=1, bindings=[int(self.input_memory), int(self.output_memory)], stream_handle=self.stream.handle)
        # Prepare buffer for output data
        host_output_data = np.empty(self.output_shape, dtype=np.float32)  # Adjust dtype and shape as necessary
        # Copy output data back to CPU
        cuda.memcpy_dtoh_async(host_output_data, self.output_memory, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        return host_output_data

    def eval(self):
        # Create CUDA events for timing GPU operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        num_warmup_iterations = 5

        # Instantiate metrics with the specified task type
        metric = MulticlassAccuracy(num_classes=5)
        precision_metric = torchmetrics.Precision(num_classes=5, average='weighted', task='multiclass')
        recall_metric = torchmetrics.Recall(num_classes=5, average='weighted', task='multiclass')
        f1_metric = torchmetrics.F1Score(num_classes=5, average='weighted', task='multiclass')

        all_accs, all_precisions, all_recalls, all_f1s = [], [], [], []
        all_losses, all_latencies, all_gpu_powers, all_gpu_energy, all_flops = [], [], [], [], []
        
        search_spaces = [self.original_graph, self.trt_context]

        # Iterate over different configurations
        for i, graph in enumerate(search_spaces):
            # Initialize lists to store metrics for each configuration
            recorded_accs = []
            latencies = []
            gpu_power_usages = []
            accs, losses = [], []
            flops = []
            
            # Iterate over batches in the training data
            for j, inputs in enumerate(self.config['data_loader']):

                # Break the loop after processing the specified number of batches
                if j >= self.config['num_batches']:
                    break

                # Unpack inputs and labels
                xs, ys = inputs

                # Instantiate and start the power monitor
                self.power_monitor.start()

                torch.cuda.empty_cache()
                
                # Record start time of the model prediction
                start.record()
                if isinstance(graph, trt.IExecutionContext):
                    preds = self.infer_trt(graph, xs)
                else:
                    preds = self.infer_mg(graph, xs)  # Run model prediction
                end.record()          # Record end time

                # Synchronize to ensure all GPU operations are finished
                torch.cuda.synchronize()

                # Calculate latency between start and end events
                latency = start.elapsed_time(end)
                latencies.append(latency)
                
                # Stop the power monitor and calculate average power
                self.power_monitor.stop()
                self.power_monitor.join()  # Ensure monitoring thread has finished
                avg_power = sum(self.power_monitor.power_readings) / len(self.power_monitor.power_readings) if self.power_monitor.power_readings else 0
                # Store the calculated average power
                gpu_power_usages.append(avg_power)
                
                # data = calculate_flops_mg_pass(graph.model)
                

                # Calculate accuracy and loss for the batch
                loss = torch.nn.functional.cross_entropy(preds, ys)
                acc = metric(preds, ys)
                accs.append(acc)
                losses.append(loss.item())
                # flops.append(data[1]['total_flops'])
                # Update torchmetrics metrics
                preds_labels = torch.argmax(preds, dim=1)
                precision_metric(preds_labels, ys)
                recall_metric(preds_labels, ys)
                f1_metric(preds_labels, ys)

            # Compute final precision, recall, and F1 for this configuration
            avg_precision = precision_metric.compute()
            avg_recall = recall_metric.compute()
            avg_f1 = f1_metric.compute()
            
            # Compute to get correct dimensions
            # avg_flops = sum(flops) / len(flops)
            
            # Reset metrics for the next configuration
            precision_metric.reset()
            recall_metric.reset()
            f1_metric.reset()
            
            if i < num_warmup_iterations:
                continue
            else:
                # Calculate and record average metrics for the current configuration
                acc_avg = sum(accs) / len(accs)
                loss_avg = sum(losses) / len(losses)
                recorded_accs.append(acc_avg)
                avg_latency = sum(latencies) / len(latencies)
                avg_gpu_power_usage = sum(gpu_power_usages) / len(gpu_power_usages)
                avg_gpu_energy_usage = (avg_gpu_power_usage / 1000) * avg_latency / (1000*3600)
                
                # Print the average metrics for the current configuration
                print(f"Configuration {i-num_warmup_iterations}:")
                print(f"Average Accuracy: {acc_avg}")
                print(f"Average Precision: {avg_precision}")
                print(f"Average Recall: {avg_recall}")
                print(f"Average F1 Score: {avg_f1}")
                print(f"Average Loss: {loss_avg}")
                print(f"Average Latency: {avg_latency} milliseconds")
                print(f"Average GPU Power Usage: {avg_gpu_power_usage} watts")
                print(f"Average GPU Energy Usage: {avg_gpu_energy_usage} kW/hr")
                # print(f"FLOPs: {avg_flops}")
                
                all_accs.append(acc_avg)
                all_precisions.append(avg_precision.item())
                all_recalls.append(avg_recall.item())
                all_f1s.append(avg_f1.item())
                all_losses.append(loss_avg)
                all_latencies.append(avg_latency)
                all_gpu_powers.append(avg_gpu_power_usage)
                all_gpu_energy.append(avg_gpu_energy_usage)
                # all_flops.append(avg_flops)


class Power_Monitor(threading.Thread):
    def __init__(self, config):
        # Initialize the NVIDIA Management Library (NVML)
        pynvml.nvmlInit()
        self.power_readings = []  # List to store power readings
        self.running = False      # Flag to control the monitoring loop
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assume using GPU 0

    def run(self):
        self.running = True
        while self.running:
            # Get current GPU power usage in milliwatts and convert to watts
            power_mW = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            power_W = power_mW / 1000.0
            self.power_readings.append(power_W)
            time.sleep(0.001)  # Wait before next reading

    def stop(self):
        self.running = False  # Stop the monitoring loop