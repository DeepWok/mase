from mase_cocotb.runner import mase_runner
from mase_cocotb.testbench import Testbench

import cocotb


# Define testbench class
class ModelTB(Testbench):
    def __init__(self, dut):
        super().__init__(dut, dut.clk, dut.rst)
        # Assign module parameters from parameter map
        # self.assign_self_params([])

        # Instantiate as many drivers as required inputs
        self.data_in_0_driver = StreamDriver(
            dut.clk, dut.data_in_0, dut.data_in_0_valid, dut.data_in_0_ready
        )

        # Instantiate as many monitors as required outputs
        self.data_out_0_monitor = StreamMonitor(
            dut.clk, dut.data_out_0, dut.data_out_0_valid, dut.data_out_0_ready
        )

    def get_random_tensor(self, shape):
        # Generate random tensors of the appropriate shape
        return torch.rand(shape)

    def generate_inputs(self):
        # Generate inputs for as many streaming interfaces as required

        # For every input to the model, generate random tensor according to its shape
        inputs = []
        inputs += get_random_tensor((1, 2))

        # Quantize each input tensor to required precision
        quantized_inputs = []
        quantized_inputs += quantize_to_int(data_in_0_inputs)

        return inputs, quantized_inputs

    def model(self, inputs):
        # Run the model with the provided inputs and return the outputs
        out = graph.model(inputs)
        return out


@cocotb.test()
async def test(dut):
    print("Running test")
    tb = ModelTB(dut)

    await tb.reset()
    tb.output_monitor.ready.value = 1
    inputs = tb.generate_inputs()
    exp_out = tb.model(inputs)

    # To do: replace with tb.load_drivers(inputs)
    for i in inputs[0]:
        tb.data_in_0_driver.append(i)

    # To do: replace with tb.load_monitors(exp_out)
    for out in exp_out:
        tb.data_out_0_monitor.expect(out)

    # To do: replace with tb.run()
    await Timer(100, units="us")
    # To do: replace with tb.monitors_done() --> for monitor, call monitor_done()
    assert tb.data_out_0_monitor.exp_queue.empty()


if __name__ == "__main__":
    mase_runner(
        module_param_list=[{}],
        extra_build_args=[],
    )
