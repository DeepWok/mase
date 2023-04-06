import os
import sys

import torch

sys.path.append("../../software")


from machop.dataset import MyDataModule
from machop.graph.passes.utils import get_input_args_generator
from machop.utils import load_pt_pl_or_pkl_checkpoint_into_pt_model

# 1. Modify-sw. Run this command under mase-tools/software to generate quantized model (pickle file)
# ./chop --modify-sw --modify-sw-config ./configs/modify-sw/integer.toml --model resnet18 --pretrained --task cls --dataset cifar10 --project resnet18_int_cifar10

# 2. Load pickle file to get quantized model
load_name = (
    "../../mase_output/resnet18_int_cifar10/software/modify-sw/modified_model.pkl"
)
resnet18_int8 = load_pt_pl_or_pkl_checkpoint_into_pt_model(
    load_name=load_name, load_type="pkl"
)
resnet18_int8.eval()

# 3. Load corresponding dataset, note that this dataset should be the same as the one in "1. modify-sw"
data_module = MyDataModule(
    name="cifar10", batch_size=4, workers=8, tokenizer=None, max_token_len=None
)
# input_args generator
input_args_generator = get_input_args_generator(
    model_name="resnet18", task="cls", data_module=data_module
)

# Then you can call "next" on input_args_generator until you run out of batches in train dataloader
with torch.no_grad():
    # fetch inputs
    input_args_0 = next(input_args_generator)
    # feed inputs to quantized model to get output
    output_0 = resnet18_int8(*input_args_0)
    print(output_0.shape)

    # next inputs
    input_args_1 = next(input_args_generator)
    output_1 = resnet18_int8(*input_args_1)
    print(output_1.shape)

    # ...
