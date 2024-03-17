import torch
import cocotb
import pytest

from torch import nn
from cocotb.triggers import Timer
from torch.autograd.function import InplaceFunction
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
    bitwidth = 8
    bias = 1
    num = 6
    high_slots = 3

    def __init__(self, threshold, samples=2, test = False):
        self.samples = samples
        self.threshold = threshold
        self.inputs = []

        for _ in range(samples):
            self.inputs.append(self.single_run())

        self.low_out, self.high_out = self.scatter_model(samples)
        

    def single_run(self):
        x = torch.rand(self.num)
        r1, r2 = 4, -4
        x = (r1 - r2) * x + r2
        x = quantize(x, self.bitwidth, self.bias)

        return x

    def scatter_model(self, samples):
        high_out = []
        low_out = []
        
        for i in range(samples):
            high_mat = []
            low_mat = []
            count_high = 0

            x = self.get_dut_input(i)

            for k in x :
                if abs(k) >= self.threshold and count_high < self.high_slots :
                    high_mat.append(k)
                    low_mat.append(0)
                    count_high += 1
                else :
                    high_mat.append(0)
                    low_mat.append(k)
                  
            low_out.append(low_mat)
            high_out.append(high_mat)

        return low_out, high_out
    

    def get_dut_parameters(self):
        return {
            "TENSOR_SIZE_DIM": self.num,
            "PRECISION": self.bitwidth,
            "HIGH_SLOTS": self.high_slots,
            "THRESHOLD": self.threshold,
        }

    def get_dut_input(self, i):
        inputs = self.inputs[i]
        shifted_integers = (inputs * (2**self.bias)).int()
        return shifted_integers.numpy().tolist()

    def get_dut_output(self, i):
        return [self.low_out[i], self.high_out[i]]



@cocotb.test()
async def test_scatter(dut):
    """Test scatter function"""
    test_case = VerificationCase(threshold=6, samples=1)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)
        y_low = y[0]
        y_high = y[1]

        print('x:', x)
        print('low_out:', y_low)
        print('high_out:', y_high)

        dut.data_in.value = x
        await Timer(2, units="ns")
    
        #for j, dutval in enumerate(dut.data_out.value):
        #    assert dutval.signed_integer == y[j]

        for j, dutval_high in enumerate(dut.o_high_precision.value):
            assert dutval_high.signed_integer == y_high[j]
            print('high:', dutval_high.signed_integer)
            
        for j, dutval_low in enumerate(dut.o_low_precision.value):
            assert dutval_low.signed_integer == y_low[j]
            print('low:', dutval_low.signed_integer)

        # assert dut.data_out.value == test_case.o_outputs_bin[0], f"output q was incorrect on the {i}th cycle"
        # print(type(dut.data_out.value))

if __name__ == "__main__":
    tb = VerificationCase(threshold=6)
    mase_runner(module_param_list=[tb.get_dut_parameters()])
    # mase_runner()