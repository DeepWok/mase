# Regression Model Generation from HLS Components

This folder contains a Python-based code generator for emitting HLS components in C.
The exploration toolchain explores an HLS component with different design constraints and returns a regression model for performance and area estimation. 

Exploring Linear2d:
```sh
python3 hls_regression.py --op=int_linear2d --mode codegen --dir dse_linear2d
cd dse_linear2d; bash run.sh
```

Exploring Softmax:
```sh
python3 hls_regression.py --op=int_softmax --mode codegen --dir dse_softmax
cd dse_softmax; bash run.sh
```

Exploring Layernorm:
```sh
python3 hls_regression.py --op=int_layernorm --mode codegen --dir dse_layernorm
cd dse_layernorm; bash run.sh
```

