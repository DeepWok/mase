# # ⚠️ Cannot run on CPU-only machine
# import mase_triton.random_bitflip
# import torch
# import transformers
# import mase_triton
# from chop.passes.module.transforms.bitflip import bitflip_module_transform_pass
# import pytest

# """
# default:
#     w_p_exp: 1.52587890625e-05
#     w_p_frac: 1.52587890625e-05
#     w_seed_exp: 0
#     w_seed_frac: 0
#     w_zero_out_t: 1.25
#     x_p_exp: 1.52587890625e-05
#     x_p_frac: 1.52587890625e-05
#     x_seed_exp: 0
#     x_seed_frac: 0
# """


# @pytest.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
# def test_bitflip_module_transform_pass():
#     model = transformers.AutoModelForCausalLM.from_pretrained("AICrossSim/clm-60m")

#     layer_name_to_cls = {name: type(m) for name, m in model.named_modules()}

#     pass_arg = {
#         "by": "regex_name",
#         "default": {
#             "x_p_exp": 1.52587890625e-05,
#             "x_p_frac": 1.52587890625e-05,
#             "x_seed_exp": 0,
#             "x_seed_frac": 0,
#             "x_zero_out_t": 1.25,
#             "w_p_exp": 1.52587890625e-05,
#             "w_p_frac": 1.52587890625e-05,
#             "w_seed_exp": 0,
#             "w_seed_frac": 0,
#             "w_zero_out_t": 1.25,
#         },
#     }

#     model_bitflip = bitflip_module_transform_pass(model, pass_arg)

#     new_layer_name_to_cls = {name: type(m) for name, m in model_bitflip.named_modules()}

#     cls_to_count = {}

#     for name, cls in new_layer_name_to_cls.items():
#         if cls != layer_name_to_cls[name]:
#             if cls.__name__ not in cls_to_count:
#                 cls_to_count[cls.__name__] = 0
#             cls_to_count[cls.__name__] += 1

#     assert (
#         cls_to_count[mase_triton.random_bitflip.layers.RandomBitFlipLinear.__name__]
#         == 155
#     )
#     print(cls_to_count)


# if __name__ == "__main__":
#     test_bitflip_module_transform_pass()
