import torch
import pytest
import cocotb
import random

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
    bitwidth = 8
    bias = 1
    num = 6
    high_slots = 3

    def __init__(self, samples=2, test = False):
        self.samples = samples
        self.low_in = []
        self.high_in = []

        for _ in range(samples):
            i1, i2 = self.single_run()
            self.low_in.append(i1)
            self.high_in.append(i2)


        self.outputs = self.gather_model(samples)


    def single_run(self):
        ya = torch.rand(self.num)
        yb = torch.rand(self.num)
      
        # Range for normalization
        r1, r2 = 4, -4
      
        # Normalize and quantize mat_a
        ya = (r1 - r2) * ya + r2
        ya = quantize(ya, self.bitwidth, self.bias)
      
        # Normalize and quantize mat_b using the same parameters
        yb = (r1 - r2) * yb + r2
        yb = quantize(yb, self.bitwidth, self.bias)

        kept = random.randint(1, self.high_slots)
        index = [random.randint(0, self.num - 1) for _ in range(kept)]

        for k in range(self.num) :
            if k in index :
                ya[k] = 0
            else :
                yb[k] = 0

        return ya, yb
    

    def gather_model(self, samples):
        outputs = []
        
        for i in range(samples):
            x_low = self.get_dut_input_0(i)
            x_high = self.get_dut_input_1(i)

            print(x_low)
            print(x_high)

            y = [x_low[k] + x_high[k] for k in range(len(x_high))]
            outputs.append(y)

        return outputs

    # Will be usefull for 2D version
    def generate_matrices(width, height, bitwidth, bias):
      # Generate random tensor for mat_a and mat_b
      mat_a = torch.rand(height, width)
      mat_b = torch.rand(height, width)
      
      # Range for normalization
      r1, r2 = 4, -4
      
      # Normalize and quantize mat_a
      mat_a = (r1 - r2) * mat_a + r2
      mat_a = quantize(mat_a, bitwidth, bias)
      
      # Normalize and quantize mat_b using the same parameters
      mat_b = (r1 - r2) * mat_b + r2
      mat_b = quantize(mat_b, bitwidth, bias)
    
      return mat_a, mat_b
    

    def get_dut_parameters(self):
        return {
            "DIM": self.num,
            "PRECISION": self.bitwidth,
        }

    def get_dut_input_0(self, i):
        inputs = self.low_in[i]
        shifted_integers = (inputs * (2**self.bias)).int()

        return shifted_integers.numpy().tolist()

    def get_dut_input_1(self, i):
        inputs = self.high_in[i]
        shifted_integers = (inputs * (2**self.bias)).int()

        return shifted_integers.numpy().tolist()

    def get_dut_output(self, i):
        return self.outputs[i]



@cocotb.test()
async def test_gather(dut):
    """Test gather function"""
    test_case = VerificationCase(samples=1)

    # set inputs outputs
    for i in range(test_case.samples):
        x_low = test_case.get_dut_input_0(i)
        x_high = test_case.get_dut_input_1(i)
        y = test_case.get_dut_output(i)

        print('x_low :', x_low)
        print('x_high :', x_high)
        print('y :', y)

        dut.mat_a.value = x_low
        dut.mat_b.value = x_high
        await Timer(2, units="ns")

        for j, output in enumerate(dut.mat_sum.value):
            #assert output.signed_integer == y[j]
            print('output:', output.signed_integer)

if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
    # mase_runner()