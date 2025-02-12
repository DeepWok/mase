## **Lab 1: Model Compression (Quantization and Pruning)**

##### 1. In Tutorial 3, you quantized every Linear layer in the model to the provided configuration. Now, explore a range of fixed point widths from 4 to 32

```
    hyper_prams=[i for i in range(4,33)]
    "linear": {
            "config": {
                "name": "integer",
                # data
                "data_in_width": hyper_pram,
                "data_in_frac_width": 4,
                # weight
                "weight_width": hyper_pram,
                "weight_frac_width": 4,
                # bias
                "bias_width": hyper_pram,
                "bias_frac_width": 4,
            }
        },
```
![lab1 1](https://tinypic.host/images/2025/01/29/lab1-1.png)

##### 2. Take your best obtained model from Task 1 and rerun the pruning procedure, this time varying the sparsity from 0.1 to 0.9

```
    best_hyper_pram=24
    hyper_prams_2=[i/10 for i in range(1,10)]
```
```
    <!-- for l1-norm -->
    pruning_config_2 = {
        "weight": {
            "sparsity": hyper_prams,
            "method": "l1-norm",
            "scope": "local",
        },
        "activation": {
            "sparsity": hyper_prams,
            "method": "l1-norm",
            "scope": "local",
        },
    }
```
```
    <!-- for random -->
    pruning_config_1 = {
        "weight": {
            "sparsity": hyper_prams,
            "method": "random",
            "scope": "local",
        },
        "activation": {
            "sparsity": hyper_prams,
            "method": "random",
            "scope": "local",
        },
    }
```

![lab1 2](https://tinypic.host/images/2025/01/29/lab1-2.png)

Noticeably, it took random norm a lot more time than the l1-norm to training. 

## **lab 2: Neural Architecture Search**

##### 1. Explore using the GridSampler and TPESampler in Optuna, plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. Plot one curve for each sampler to compare their performance.

We should use the same name for search_space with  trial.suggest_categorical to avoid mistakes for grid sampler.

```
<!-- for grid sampler -->
search_space = {
    "num_layers": [2, 4, 8],
    "num_heads": [2, 4, 8, 16],
    "hidden_size": [128, 192, 256, 384, 512],
    "intermediate_size": [512, 768, 1024, 1536, 2048],
    "linear_layer_choices": [0, 1],  
}
```
```
def construct_model(trial):
    config = AutoConfig.from_pretrained(checkpoint)
    for param in [
        "num_layers",
        "num_heads",
        "hidden_size",
        "intermediate_size",
    ]:
        chosen_idx = trial.suggest_categorical(param, search_space[param])
        setattr(config, param, chosen_idx)

    trial_model = AutoModelForSequenceClassification.from_config(config)
    linear_layer_mapping = {0: nn.Linear, 1: Identity}
    for name, layer in trial_model.named_modules():
        if isinstance(layer, nn.Linear) and layer.in_features == layer.out_features:
            new_layer_cls = trial.suggest_categorical(
                "linear_layer_choices",
                search_space["linear_layer_choices"],
            )
            new_layer_cls=linear_layer_mapping[new_layer_cls]
            if new_layer_cls == nn.Linear:
                continue
            elif new_layer_cls == Identity:
                new_layer = Identity()
                deepsetattr(trial_model, name, new_layer)
            else:
                raise ValueError(f"Unknown layer type: {new_layer_cls}")

    return trial_model
```

![lab2 1](https://tinypic.host/images/2025/01/30/lab2-1.png)

##### 2. In the objective function, after the model is constructed and trained for some iterations, call the CompressionPipeline to quantize and prune the model, then continue training for a few more epochs. Use the sampler that yielded the best results in Task 1 to run the compression-aware search.

In task, lab2, the gridsampler is better at the beginning. 

The model suffers from compressing, but will get better results than original after training agian.
###### 2.1 TPESampler

![lab2 plus](https://tinypic.host/images/2025/01/29/lab2-plus.png)

###### 2.2 GridSampler

![lab2 2](https://tinypic.host/images/2025/01/30/lab2-2.png)

## **lab 3 Mixed Precision Search**

##### 3.1 Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis.
The normal nn.linear has better results in trails, reaching the accuracy of over 0.85. Most other linear layers added to search_space reaches an accuracy around 0.5. But there is still have the chance for the model to choose nn.linear, which explains why most of the lines in the graph reaches over 0.85 in the end.
![lab3 1-new](https://tinypic.host/images/2025/02/02/image.png)

##### 3.2 Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. Plot one curve for each precision to compare their performance

![image](https://tinypic.host/images/2025/02/10/image74ced402de0cf8ea.png)
## **lab 4: (Software Stream) Performance Engineering**

##### 4.1 In the first part of Lab 4 (torch.compile), we did not really observe real run-time speedups with torch.compile.

###### 4.1.1 On CPU

| Run  | Original model (s) | Optimized model (s) |
|-------|--------------------|--------------------|
| 1     | 2.4651            | 6.4006            |
| 2     | 2.6122            | 1.6296            |
| 3     | 2.5208            | 1.8173            |
| 4     | 2.1772            | 1.5918            |


###### 4.1.2 On CUDA

| Run  | Original model (s) | Optimized model (s) |
|-------|--------------------|--------------------|
| 1     | 1.9287            | 11.1671           |
| 2     | 0.1113            | 0.0667            |
| 3     | 0.0750            | 0.0656            |
| 4     | 0.0761            | 0.0666            |


**Why torch.compile may be slower running on GPUs than CPUs**
- From the table above it is clear that both CPUs and GPUs need a relatively long time in the first run.
- Also, in that first run, the original model always beat the optimized model.
- However, it basically took GPUs much less time in the following trails. In the meantime, the optimized model beat the original models(for around 10 times less in time/s).
## Possible Reasons Why `torch.compile` Might Slow Down a Model

- 1. Compilation Overhead  
The first execution of an optimized model requires additional time for compilation. `torch.compile` transforms the model into an optimized computation graph, leading to increased execution time during the initial run.

- 2. Suboptimal Kernel Selection  
The backend `TorchInductor` is responsible for generating efficient kernels. However, it may not always choose the most optimal kernel, leading to performance degradation, especially on GPUs.

- 3. JIT Compilation Effects  
PyTorch’s Just-In-Time (JIT) compiler optimizes execution dynamically. Without `torch.compile`, PyTorch may have already cached certain computations, whereas `torch.compile` reprocesses them, adding extra overhead.

- 4. Graph Optimization Overhead  
The optimization process may introduce additional computation steps, particularly with AutoGraph and graph fusion techniques. This can lead to unintended performance losses instead of gains.

- 5. CUDA Initialization Delays  
On GPUs, CUDA requires initialization and kernel loading, which contributes to the observed slowdown during the first execution. However, subsequent executions typically show performance improvements.



##### 4.2 In the second part of Lab 4 (kernel fusion), we looked at a fused SDPA kernel

###### 4.2.1 On CPU

| Run  | Original model (s) | Optimized model (s) |
|-------|--------------------|--------------------|
| 1     | 0.0141            | 0.0056            |
| 2     | 0.0136            | 0.0061            |
| 3     | 0.0134            | 0.0064            |
| 4     | 0.0144            | 0.0068            |


###### 4.2.2 On CUDA

| Run  | Original model (s) | Optimized model (s) |
|-------|--------------------|--------------------|
| 1     | 0.0124            | 0.0010            |
| 2     | 0.0134            | 0.0031            |
| 3     | 0.0128            | 0.0008            |
| 4     | 0.0139            | 0.0031            |

It is clear from the table that on both GPUs and CPUs,  optimized model performs better than the original models, indicating kernel fusion is a reliable way to accelerate inference.


##### 4.3 In the third part of lab4 (Custom kernel), we go through how to write MXINT8 dequantization kernel and bind it to Python.

###### 4.3.1 How does MXINT8 benefit custom hardware if both the activation and weights in a linear layer are quantized to MXINT8?

MXINT8 reduces both memory bandwidth and computation costs. Since integer operations are generally faster and require less power than floating-point operations, using MXINT8 can significantly improve inference speed and efficiency on specialized hardware (e.g., TPUs, FPGAs, or custom AI accelerators). Moreover, reduced memory footprint allows more data to fit into cache, reducing memory access latency and improving overall throughput.

###### 4.3.2 What is the purpose of the variable `dont_need_abs` and `bias` in the C++ for loop?

- `dont_need_abs`: This variable is typically used to bypass absolute value operations if they are unnecessary for the given computation, optimizing performance by skipping redundant calculations.
- `bias`: In dequantization, bias helps adjust the quantized values back to their original floating-point range, ensuring numerical stability and reducing quantization-induced errors.

###### 4.3.3 How does `cta_tiler` partition data for copying to shared memory in CUDA kernel? How does `layout_sX` partition threads in a threadblock for computation?

- `cta_tiler`: This mechanism splits global memory data into tiles that fit into shared memory, ensuring coalesced memory access and improving memory efficiency by reducing global memory transactions.
- `layout_sX`: This structure defines how threads within a threadblock are mapped to specific computation units, optimizing parallelism and minimizing thread divergence for better GPU performance.

###### 4.3.4 Why is the saved GPU memory not exactly (32 - (4+8/32))/32 = 86.7% of the FP32 model?

The theoretical memory savings assume perfect quantization efficiency, but in practice, overhead such as padding, memory alignment, and additional scaling factors reduces the actual savings. Furthermore, some GPU architectures require extra storage for mixed-precision computation, leading to a slightly different memory footprint than the calculated value.
