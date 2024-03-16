import torch

from torch import nn
from torch.autograd.function import InplaceFunction

import cocotb

from cocotb.triggers import Timer

import pytest
import cocotb

from mase_cocotb.runner import mase_runner

pytestmark = pytest.mark.simulator_required




# fixed-point quantization with a bias
def quantize(x, bits, bias):  # bits = 32
    """Do linear quantization to input according to a scale and number of bits"""
    thresh = 2 ** (bits - 1)
    scale = 2**bias
    return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)


class VerificationCase:
    bitwidth = 4
    bias = 1
      
    x = 4
    y = 4

    high_slots = 1
    o_low_precision = []
    o_high_precision = []
    o_outputs_bin = []

    def __init__(self, samples=2,test = False):
        # self.m = nn.ReLU()
        self.inputs, self.outputs = [], []
        for _ in range(samples):
            i1,i2, o= self.single_run()

            self.inputs.append([i1,i2])
            self.outputs.append(o)

        # self.o_outputs_bin, self.o_low_precision, self.o_high_precision = self.scatter_model(samples)
        # print('outputs_bin',self.o_outputs_bin)
        self.samples = samples

    def single_run(width, height, bitwidth, bias):
      mat_a, mat_b = generate_matrices(width, height, bitwidth, bias)
      mat_sum = mat_a + mat_b
      return mat_a, mat_b, mat_sum
    

    def gather_model(self, samples):
        outputs = []
        high_out= []
        low_out = []
        
        for i in range(samples):
          x = self.get_dut_input(i)
        #   print('quant_input_val',x)
          x_bin = self.to_twos_complement(x)
        #   print('bin_quant_input_val',x_bin)
          outputs.append(x_bin)

        return outputs,high_out,low_ou

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
            "DIM_X": self.x,
            "DIM_Y": self.y,
            "PRECISION": self.bitwidth,
            "HIGH_SLOTS": self.high_slots,

        }

    # def get_dut_input(self, i):
    #     return self.inputs[i]

    # def get_dut_output(self, i):

    #     return self.outputs[i].int


    def get_dut_input_0(self, i):
        inputs = self.inputs[i][0]
        shifted_integers = (inputs * (2**self.bias)).int()
        # if(self.num == 1):
        #     print("input",shifted_integers)
        #     return shifted_integers
        return shifted_integers.numpy().tolist()


    def get_dut_input_1(self, i):
        inputs = self.inputs[i][1]
        shifted_integers = (inputs * (2**self.bias)).int()
        # if(self.num == 1):
        #     print("input",shifted_integers)
        #     return shifted_integers
        return shifted_integers.numpy().tolist()

    def get_dut_output(self, i):
        outputs = self.outputs[i]
        shifted_integers = (outputs * (2**self.bias)).int()
        # if(self.num == 1):
        #     return shifted_integers.int()
        return shifted_integers.numpy().tolist()


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
    """Test integer based Relu"""
    test_case = VerificationCase(samples=10)

    # set inputs outputs
    for i in range(test_case.samples):
        x1 = test_case.get_dut_input_0(i)
        x2 = test_case.get_dut_input_1(i)

        y = test_case.get_dut_output(i)
        print('x:',x)

        dut.data_in.value = x
        await Timer(2, units="ns")

    
        for i, dutval in enumerate(dut.data_out.value):
          assert dutval.signed_integer == y[i]
        for i, dutval_high in enumerate(dut.o_high_precision.value):
            print('high:',dutval_high.signed_integer)
        for i, dutval_low in enumerate(dut.o_high_precision.value):
            print('low:',dutval_low.signed_integer)
        # assert dut.data_out.value == test_case.o_outputs_bin[0], f"output q was incorrect on the {i}th cycle"
        # print(type(dut.data_out.value))

if __name__ == "__main__":
    tb = VerificationCase()
    mase_runner(module_param_list=[tb.get_dut_parameters()])
    # mase_runner()