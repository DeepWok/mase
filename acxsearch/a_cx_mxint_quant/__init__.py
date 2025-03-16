
DEIT_TINY_IMAGENET_ACC = 0.72132
DEIT_TINY_IMAGENET_ACC_100ITER = 0.792
DEIT_SMALL_IMAGENET_ACC = 0.0 # TODO: 
DEIT_BASE_IMAGENET_ACC = 0.0 # TODO: 

# This is the default quant config for the MXIntQuant class
# It is used to quantize the model during the search process
# The default settings should be lossless (Tested on DeiT-tiny)

exponent_width = 8
quant_config = {
    "by": "type",
    "gelu": {
        "config": {
            "data_in_width": 8,
            "data_in_exponent_width": exponent_width,
            "data_in_parallelism": (1, 32),
            "data_out_width": 8,
            "data_out_exponent_width": exponent_width,
            "data_out_parallelism": (1, 32),
            "enable_internal_width": True,
            "hash_in_int_width": 16,
            "hash_in_frac_width": 16,
            "hash_out_int_width": 16,
            "hash_out_frac_width": 16,
            "hash_in_int_width": 16,
            "hash_in_frac_width": 16,
            "hash_out_int_width": 16,
            "hash_out_frac_width": 16,
        }
    },
}
