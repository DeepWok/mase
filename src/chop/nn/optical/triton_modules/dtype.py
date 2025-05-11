import torch
import triton.language as tl


TORCH_DTYPE_TO_TRITON = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
    torch.int16: tl.int16,
    torch.uint16: tl.uint16,
    torch.int32: tl.int32,
    torch.uint32: tl.uint32,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
}
