import os
import sys

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
sys.path.append("../../software")
import sys

import torch
from machop.modify.quantizers.quantizers_for_hw import integer_quantizer_for_hw
from machop.sim_hw.input_iterator import InputIterator
from machop.utils import load_pt_pl_or_pkl_checkpoint_into_pt_model

"""
1. Modify-sw. Run this command under mase-tools/software to generate quantized model (pickle file)
   ./chop --modify-sw --modify-sw-config ./configs/modify-sw/integer.toml --model toy --task cls --dataset cifar10 --project toy_int8_cifar10
   Pretrained weight is not available for Toy, but other models like resnet18 have pretrained weights.
   Thus it is possible to load pretrained weight into resnet18 before quantization, like the following.
   ./chop --modify-sw --modify-sw-config ./configs/modify-sw/integer.toml --model resnet18 --task cls --dataset cifar10 --project resnet18_int8_cifar10
"""

"""
2. Load the saved quantized model
"""
load_name = "../../mase_output/toy_int8_cifar10/software/modify-sw/modified_model.pkl"
toy_int8 = load_pt_pl_or_pkl_checkpoint_into_pt_model(
    load_name=load_name, load_type="pkl"
)
toy_int8.eval()

"""
3. Create a input iterator for fetching inputs for sw model and hw simulation.
   - parameter `quant_fn` is used to quantize the dataset sample
   - `integer_quantizer_for_hw` is like the sw version `integer_quantizer`, except that `integer_quantizer_for_hw` does not do de-quantization.
     So `integer_quantizer_for_hw` returns signed integers
   - `integer_quantizer_for_hw` is defined here: mase-tools/software/machop/modify/quantizers/quantizers_for_hw.py
   - `quant_config` is required for `integer_quantizer_for_hw`
"""

input_args_generator = InputIterator(
    model_name="toy",
    task="cls",
    dataset_name="cifar10",
    quant_fn=integer_quantizer_for_hw,
    quant_config_kwargs={"width": 8, "frac_width": 7},
)

"""
4. Then you can call "next" on input_args_generator until you run out of batches in train dataloader
   - next(input_args_generator) returns two lists: input_args_for_sw, input_args_for_hw.
   - For vision models on cls task, returned `sw_input_args`/ `hw_input_args` only has a single element, i.e., normalized input image
"""
with torch.no_grad():
    # call `next` on input_args_generator to fetch inputs
    sw_input_args_0, hw_input_args_0 = next(input_args_generator)
    # feed inputs to quantized model to get output
    output_0 = toy_int8(*sw_input_args_0)
    print(sw_input_args_0[0][0, 0, 0, :4])
    print(hw_input_args_0[0][0, 0, 0, :4])

    # next inputs
    sw_input_args_1, hw_input_args_1 = next(input_args_generator)
    output_1 = toy_int8(*sw_input_args_1)

    # ...
