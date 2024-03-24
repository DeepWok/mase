import torch
import pytest
import cocotb

import torch.nn as nn
from torch.autograd.function import InplaceFunction
from cocotb.triggers import Timer
from mase_cocotb.runner import mase_runner

pytestmark = pytest.mark.simulator_required


# snippets
class MyClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class MyRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# wrap through module to make it a function
my_clamp = MyClamp.apply
my_round = MyRound.apply

# fixed-point quantization with a bias
def quantize(x, bits, bias):  # bits = 32
    """Do linear quantization to input according to a scale and number of bits"""
    thresh = 2 ** (bits - 1)
    scale = 2**bias
    return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)


class VerificationCase:
    weight_dim = [6,6]
    bitwidth = 8
    reduced_bitwidth = 4
    bias = 1
    num = 6
    high_slots = 3
    threshold = 6

    def __init__(self, samples=2, test = False):
        self.samples = samples
        self.inputs = []
        self.weights = []

        for _ in range(samples):
            self.inputs.append(self.single_run())
            self.weights.append(self.generate_weights())

        self.outputs = self.LLMint_model(samples)


    def single_run(self):
        x = torch.rand(self.num)
        r1, r2 = 4, -4
        x = (r1 - r2) * x + r2
        x = quantize(x, self.bitwidth, self.bias)

        return x
    
    def scatter(self, x):
        
        high_mat = []
        low_mat = []
        count_high = 0
                    
        for k in reversed(x) :
            if abs(k) > self.threshold and count_high < self.high_slots :
                high_mat.append(k)
                low_mat.append(0)
                count_high += 1
            else :
                high_mat.append(0)
                low_mat.append(k)

        low_mat.reverse()
        high_mat.reverse()

        return low_mat, high_mat
    

    def gather(self, x_low, x_high):
        return [x_low[k] + x_high[k] for k in range(len(x_high))]
    
    # LLMint model is the combination of scatter, 2 linear layers and gather
    # Scatter separates the input into high and low precision values based on a threshold (padded with zeros)
    # Low precision values are quantized and passed through a low precision linear layer
    # High precision values are passed through a high precision linear layer
    # The outputs of the two linear layers are then combined using gather
    def LLMint_model(self, samples):
        outputs = []

        for i in range(samples):
            x = self.get_dut_input(i)
            w = self.get_dut_weights(i)
            x_low, x_high = self.scatter(x)

            x_low = torch.tensor(x_low)
            x_high = torch.tensor(x_high)
            w = torch.tensor(w)

            x_low_q = quantize(x_low, self.reduced_bitwidth, self.bias)
            x_high_q = quantize(x_high, self.bitwidth, self.bias)
            w = quantize(w, self.bitwidth, self.bias)
            w_q = quantize(w, self.reduced_bitwidth, self.bias)

            linear_low = nn.Linear(self.weight_dim[0], self.weight_dim[1], bias=False)
            linear_high = nn.Linear(self.weight_dim[0], self.weight_dim[1], bias=False)
            linear_low.weight.data = w_q
            linear_high.weight.data = w

            x_low_o = linear_low(x_low_q).tolist()
            x_high_o = linear_high(x_high_q).tolist()

            outputs.append(self.gather(x_low_o, x_high_o))

        return outputs


    def generate_weights(self):
      # Generate random tensor for mat_a and mat_b
      weights = torch.rand(self.weight_dim[0], self.weight_dim[1])
      
      # normalization and quantization
      r1, r2 = 4, -4
      weights = (r1 - r2) * weights + r2
      weights = quantize(weights, self.bitwidth, self.bias)
    
      return weights
    

    def get_dut_parameters(self):
        return {
            "ORIGINAL_PRECISION": self.bitwidth,
            "REDUCED_PRECISION": self.reduced_bitwidth,
            "TENSOR_SIZE_DIM": self.num,
            "WEIGHT_DIM_0": self.weight_dim[0],
            "WEIGHT_DIM_1": self.weight_dim[1],
            "HIGH_SLOTS": self.high_slots,
            "THRESHOLD": self.threshold,
        }

    def get_dut_input(self, i):
        inputs = self.inputs[i]
        shifted_integers = (inputs * (2**self.bias)).int()
        return shifted_integers.numpy().tolist()
    
    def get_dut_weights(self, i):
        weights = self.weights[i]
        shifted_integers = (weights * (2**self.bias)).int()
        return shifted_integers.numpy().tolist()

    def get_dut_output(self, i):
        outputs = self.outputs[i]
        return outputs



@cocotb.test()
async def test_LLMint(dut):
    """Test LLmint module"""
    test_case = VerificationCase(samples=1)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        weights = test_case.get_dut_weights(i)
        flattened_weights = [item for sublist in weights for item in sublist]
        y = test_case.get_dut_output(i)

        print('x :', x)
        print('weights :', flattened_weights)
        print('y :', y)

        dut.data_in.value = x
        dut.weights.value = flattened_weights
        await Timer(2, units="ns")

        print('output dut :', dut.data_out.value)

        for j, output in enumerate(dut.data_out.value):
            assert output.signed_integer == y[j]
            print('output:', output.signed_integer)

if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
    # mase_runner()