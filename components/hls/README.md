# Regression Model Generation from HLS Components

This folder contains a Python-based code generator for emitting HLS components in C.
The exploration toolchain explores an HLS component with different design constraints and returns a regression model for performance and area estimation. 

Exploring Linear2d:
```sh
python3 hls_regression.py --op=int_linear2d --mode codegen --dir dse_linear2d
cd dse_linear2d; bash run.sh; cd ..
python3 hls_regression.py --op=int_linear2d --mode report --dir dse_linear2d
```

Exploring Matmul:
```sh
python3 hls_regression.py --op=int_matmul --mode codegen --dir dse_matmul
cd dse_matmul; bash run.sh
```

Exploring Softmax:
```sh
python3 hls_regression.py --op=int_softmax --mode codegen --dir dse_softmax
cd dse_softmax; bash run.sh; cd .. 
python3 hls_regression.py --op=int_softmax --mode report --dir dse_softmax
```

Exploring Layernorm:
```sh
python3 hls_regression.py --op=int_layernorm --mode codegen --dir dse_layernorm
cd dse_layernorm; bash run.sh; cd ..
python3 hls_regression.py --op=int_layernorm --mode report --dir dse_layernorm
```

Exploring Element-wise mult or add:
```sh
python3 hls_regression.py --op=int_mult --mode codegen --dir dse_mult
cd dse_mult; bash run.sh; cd ..
python3 hls_regression.py --op=int_mult --mode report --dir dse_mult
python3 hls_regression.py --op=int_add --mode codegen --dir dse_add
cd dse_add; bash run.sh; cd ..
python3 hls_regression.py --op=int_add --mode report --dir dse_add
```

Exploring ReLU:
```sh
python3 hls_regression.py --op=int_relu --mode codegen --dir dse_relu
cd dse_relu; bash run.sh; cd ..
python3 hls_regression.py --op=int_relu --mode report --dir dse_relu
```

Exploring SiLU:
```sh
python3 hls_regression.py --op=int_silu --mode codegen --dir dse_silu
cd dse_silu; bash run.sh; cd ..
python3 hls_regression.py --op=int_silu --mode report --dir dse_silu
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


