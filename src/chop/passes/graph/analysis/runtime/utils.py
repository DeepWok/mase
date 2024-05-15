import threading
import pynvml
import time


class PowerMonitor(threading.Thread):
    def __init__(self, config):
        super().__init__()  # Call the initializer of the base class, threading.Thread
        # Initialize the NVIDIA Management Library (NVML)
        pynvml.nvmlInit()
        self.power_readings = []  # List to store power readings
        self.running = False  # Flag to control the monitoring loop
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


def get_execution_provider(accelerator):
    EP_list = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return (
        ["CUDAExecutionProvider"] if accelerator == "cuda" else ["CPUExecutionProvider"]
    )
    # return ["TensorrtExecutionProvider"]
