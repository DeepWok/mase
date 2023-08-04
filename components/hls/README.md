# Regression Model Generation from HLS Components

This folder contains a Python-based code generator for emitting HLS components in C.
The exploration toolchain explores an HLS component with different design constraints and returns a regression model for performance and area estimation. 

Exploring int Linear2d:
```sh
python3 hls_regression.py --op=int_linear2d --mode codegen --dir dse_int_linear2d
cd dse_int_linear2d; bash run.sh; cd ..
python3 hls_regression.py --op=int_linear2d --mode report --dir dse_int_linear2d
```

Exploring Matmul:
```sh
python3 hls_regression.py --op=int_matmul --mode codegen --dir dse_int_matmul
cd dse_int_matmul; bash run.sh
```

Exploring Softmax:
```sh
python3 hls_regression.py --op=int_softmax --mode codegen --dir dse_int_softmax
cd dse_int_softmax; bash run.sh; cd .. 
python3 hls_regression.py --op=int_softmax --mode report --dir dse_int_softmax
```

Exploring RMSNorm:
```sh
python3 hls_regression.py --op=int_rmsnorm --mode codegen --dir dse_int_rmsnorm
cd dse_rmsnorm; bash run.sh; cd .. 
python3 hls_regression.py --op=int_rmsnorm --mode report --dir dse_int_rmsnorm
```

Exploring RoPE:
```sh
python3 hls_regression.py --op=int_rope --mode codegen --dir dse_int_rope
cd dse_rope; bash run.sh; cd .. 
python3 hls_regression.py --op=int_rope --mode report --dir dse_int_rope
```

Exploring Layernorm:
```sh
python3 hls_regression.py --op=int_layernorm --mode codegen --dir dse_int_layernorm
cd dse_int_layernorm; bash run.sh; cd ..
python3 hls_regression.py --op=int_layernorm --mode report --dir dse_int_layernorm
```

Exploring Element-wise int mult or add:
```sh
python3 hls_regression.py --op=int_mult --mode codegen --dir dse_int_mult
cd dse_int_mult; bash run.sh; cd ..
python3 hls_regression.py --op=int_mult --mode report --dir dse_int_mult
python3 hls_regression.py --op=int_add --mode codegen --dir dse_int_add
cd dse_int_add; bash run.sh; cd ..
python3 hls_regression.py --op=int_add --mode report --dir dse_int_add
```

Exploring ReLU:
```sh
python3 hls_regression.py --op=int_relu --mode codegen --dir dse_int_relu
cd dse_int_relu; bash run.sh; cd ..
python3 hls_regression.py --op=int_relu --mode report --dir dse_int_relu
```

Exploring SiLU:
```sh
python3 hls_regression.py --op=int_silu --mode codegen --dir dse_int_silu
cd dse_int_silu; bash run.sh; cd ..
python3 hls_regression.py --op=int_silu --mode report --dir dse_int_silu
```

Exploring transpose:
```sh
python3 hls_regression.py --op=int_transpose --mode codegen --dir dse_transpose
cd dse_transpose; bash run.sh; cd ..
python3 hls_regression.py --op=int_transpose --mode report --dir dse_transpose
```

Exploring fork:
```sh
python3 hls_regression.py --op=fork --mode codegen --dir dse_fork
cd dse_fork; bash run.sh; cd ..
python3 hls_regression.py --op=fork --mode report --dir dse_fork
```

Exploring buffer:
```sh
python3 hls_regression.py --op=buffer --mode codegen --dir dse_buffer
cd dse_buffer; bash run.sh; cd ..
python3 hls_regression.py --op=buffer --mode report --dir dse_buffer
```

Exploring Element-wise bfp mult or add:
```sh
python3 hls_regression.py --op=bfp_mult --mode codegen --dir dse_bfp_mult
cd dse_bfp_mult; bash run.sh; cd ..
python3 hls_regression.py --op=bfp_mult --mode report --dir dse_bfp_mult
python3 hls_regression.py --op=bfp_add --mode codegen --dir dse_bfp_add
cd dse_bfp_add; bash run.sh; cd ..
python3 hls_regression.py --op=bfp_add --mode report --dir dse_bfp_add
```

Exploring bfp Linear2d:
```sh
python3 hls_regression.py --op=bfp_linear2d --mode codegen --dir dse_bfp_linear2d
cd dse_bfp_linear2d; bash run.sh; cd ..
python3 hls_regression.py --op=bfp_linear2d --mode report --dir dse_bfp_linear2d
```


