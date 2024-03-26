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


class ScatterVerificationCase:
    bias = 1
    samples = 1

    def __init__(self, dut, samples=20, test = False):
        self.get_dut_parameters(dut)
        self.samples = samples
        self.inputs = []
        # print('CASE: bitwidth', self.bitwidth, 'tensor size',self.num, 'high slots',self.high_slots, ' THRESHOLD')
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
            low_out.append(low_mat)
            high_out.append(high_mat)

        return low_out, high_out
    

    def make_dut_parameters(self):


        return {
            "TENSOR_SIZE_DIM": self.num,
            "PRECISION": self.bitwidth,
            "HIGH_SLOTS": self.high_slots,
            "THRESHOLD": self.threshold,
        }


    def get_dut_parameters(self,dut):
        self.num = dut.TENSOR_SIZE_DIM.value
        self.bitwidth = dut.PRECISION.value
        self.high_slots = dut.HIGH_SLOTS.value
        self.threshold = dut.THRESHOLD.value


    def get_dut_input(self, i):
        inputs = self.inputs[i]
        shifted_integers = (inputs * (2**self.bias)).int()
        return shifted_integers.numpy().tolist()

    def get_dut_output(self, i):
        return [self.low_out[i], self.high_out[i]]

    

@cocotb.test()
async def test_scatter(dut):
    """Test scatter function"""


    test_case = ScatterVerificationCase(dut,samples=1)


    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)
        y_low = y[0]
        y_high = y[1]


       
        dut.data_in.value = x
        await Timer(1, units="ns")

            

        # print('HIGH_SLOTS',dut.HIGH_SLOTS.value)
        # print('x:', x)
        # print('low_out:', y_low)
        # print('dutval low', dut.o_low_precision.value)

        # print('high_out:', y_high)
        # print('dutval high', dut.o_high_precision.value)
    
        #for j, dutval in enumerate(dut.data_out.value):
        #    assert dutval.signed_integer == y[j]

        for j, dutval_high in enumerate(dut.o_high_precision.value):

            assert dutval_high.signed_integer == y_high[j]
            # print('dutval_high,y_high',dutval_high.signed_integer,y_high[j])

            # print('high:', dutval_high.signed_integer)
            
        for j, dutval_low in enumerate(dut.o_low_precision.value):
            assert dutval_low.signed_integer == y_low[j]
            # print('dutval_low,y_low',dutval_low.signed_integer,y_low[j])

        # assert dut.data_out.value == test_case.o_outputs_bin[0], f"output q was incorrect on the {i}th cycle"
        # print(type(dut.data_out.value))

#### TEST 39 : {'PRECISION': 32, 'TENSOR_SIZE_DIM': 32, 'HIGH_SLOTS': 3, 'THRESHOLD': 1, 'DESIGN': 1}
#### TEST 40 : {'PRECISION': 32, 'TENSOR_SIZE_DIM': 32, 'HIGH_SLOTS': 4, 'THRESHOLD': 1, 'DESIGN': 1}
def create_params(precision_values=[32], #[8, 16, 32, 64],
                        tensor_size_dim_values= [4], #[ 4, 8, 16, 32],
                        high_slots_values= [1, 2, 3, 4, 5],
                        threshold_values= range(1, 2, 2)):
        module_param_list = []

        for precision in precision_values:
            for tensor_size_dim in tensor_size_dim_values:
                for high_slots in high_slots_values:
                    if high_slots < tensor_size_dim // 2:  # Ensure high_slots is less than half of tensor_size_dim
                        for threshold in threshold_values:
                            module_param = {
                                "PRECISION": precision,
                                "TENSOR_SIZE_DIM": tensor_size_dim,
                                "HIGH_SLOTS": high_slots,
                                "THRESHOLD": threshold,
                                "DESIGN": 1
                            }
                            module_param_list.append(module_param)


                    # Display the first few dictionaries to check
        for i in range(len(module_param_list)):
            print(module_param_list[i])

        return module_param_list


if __name__ == "__main__":

    module_param_list = create_params()
    mase_runner(module_param_list, extra_build_args= ["-Wno-style"], trace =False)


    # mase_runner(module_param_list=[tb.make_dut_parameters()])


    # mase_runner()

    