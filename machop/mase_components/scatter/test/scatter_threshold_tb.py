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
    bitwidth = 4
    bias = 1
    num = 6
    high_slots = 2

    def __init__(self, threshold, samples=2, test = False):
        self.samples = samples
        self.threshold = threshold
        self.inputs = []

        for _ in range(samples):
            self.inputs.append(self.single_run())

        self.low_out, self.high_out = self.scatter_model(samples)
        

    def single_run(self):
        xs = torch.rand(self.num) * 10
        
        #r1, r2 = 4, -4
        #xs = (r1 - r2) * xs + r2
        # 8-bit, (5, 3)
        #xs = quantize(xs, self.bitwidth, self.bias)
        # if(self.num == 1):
        #    return xs[0], xs[0]

        return xs

    def scatter_model(self, samples):
        high_out = []
        low_out = []
        
        for i in range(samples):
            high_mat = []
            low_mat = []
            count_high = 0

            x = self.get_dut_input(i)

            for k in x :
                if k >= self.threshold and count_high < 3 :
                    high_mat.append(k)
                    low_mat.append(0.0)
                    count_high += 1
                else :
                    high_mat.append(0.0)
                    low_mat.append(k)
                  
            low_out.append(low_mat)
            high_out.append(high_mat)

        return low_out, high_out
    

    def get_dut_parameters(self):
        return {
            "DATA_IN_0_TENSOR_SIZE_DIM_0": self.num,
            "DATA_IN_0_PRECISION_0": self.bitwidth,
            "DATA_OUT_0_PRECISION_0": self.bitwidth,
            "HIGH_SLOTS": self.high_slots,
        }


    def get_dut_input(self, i):
        inputs = self.inputs[i]
        shifted_integers = (inputs * (2**self.bias)).int()
        # if(self.num == 1):
        #     print("input",shifted_integers)
        #     return shifted_integers
        return shifted_integers.numpy().tolist()

    def get_dut_output(self, i):
        low_outputs = self.low_out[i]
        high_outputs = self.high_out[i]

        low_shifted_integers = [(k * (2**self.bias)) for k in low_outputs]
        high_shifted_integers = [(k * (2**self.bias)) for k in high_outputs]

        outputs = [low_shifted_integers, high_shifted_integers]

        return outputs


    def to_twos_complement(self, integers):
        return [format((1 << self.bitwidth) + x if x < 0 else x, f'0{self.bitwidth}b') for x in integers]


    def int_to_signed_magnitude_binary(self,number):
      # Determine the sign bit (0 for positive, 1 for negative)
      if number >= 0:
          sign_bit = '0'
      else:
          sign_bit = '1'
          number = -number  # Make the number positive for binary conversion

      # Convert the absolute value to binary
      binary_representation = bin(number)[2:]  # [2:] to remove the '0b' prefix

      # Ensure the binary representation fits the desired total length, including the sign bit
      if len(binary_representation) < (self.bitwidth - 1):
          # Prepend zeros to reach the desired length
          binary_representation = binary_representation.rjust(self.bitwidth - 1, '0')
      elif len(binary_representation) > (self.bitwidth - 1):
          raise ValueError("The number is too large to fit in the specified total length")

      # Combine the sign bit with the binary representation
      signed_magnitude_binary = sign_bit + binary_representation
      
      return signed_magnitude_binary


    def int_list_to_signed_magnitude_binary(self,int_list):    
      binary_list = [self.int_to_signed_magnitude_binary(number) for number in int_list]
      return binary_list



@cocotb.test()
async def test_scatter(dut):
    """Test scatter function"""
    test_case = VerificationCase(threshold=6, samples=1)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)
        print('x:', x)
        print('low_out:', y[0])
        print('high_out:', y[1])

        dut.data_in.value = x
        await Timer(2, units="ns")

    
        for i, dutval in enumerate(dut.data_out.value):
          assert dutval.signed_integer == y[i]
        for i, dutval_high in enumerate(dut.o_high_precision.value):
            print('high:',dutval_high.signed_integer)
            
        for i, dutval_low in enumerate(dut.o_low_precision.value):
            print('low:',dutval_low.signed_integer)
        # assert dut.data_out.value == test_case.o_outputs_bin[0], f"output q was incorrect on the {i}th cycle"
        # print(type(dut.data_out.value))

if __name__ == "__main__":
    tb = VerificationCase(threshold=6)
    mase_runner(module_param_list=[tb.get_dut_parameters()])
    # mase_runner()