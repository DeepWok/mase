## Q1 — Why can `torch.compile` appear slower, and when does it actually help?

In early runs of this experiment, the *compiled* model sometimes appeared **slower** than the eager (uncompiled) model. This behaviour is expected and can be explained by how `torch.compile` works internally, how benchmarking is performed, and the characteristics of the available hardware.

---

## How `torch.compile` works (conceptual recap)

`torch.compile` is an **automatic optimization pipeline**, not a single optimizer. It consists of three main components:

### 1. TorchDynamo
- Captures the PyTorch program by intercepting Python bytecode.
- Extracts a graph-level representation of the model execution.
- Enables graph-level reasoning while preserving Python semantics.

🔗 https://github.com/pytorch/pytorch/tree/main/torch/_dynamo

---

### 2. TorchInductor
- Lowers the captured graph to optimized kernels.
- Performs **operator fusion**, **layout planning**, and **kernel specialization**.
- Uses **Triton** to generate high-performance GPU kernels.

🔗 https://github.com/pytorch/pytorch/tree/main/torch/_inductor  
🔗 https://github.com/pytorch/pytorch/tree/main/torch/_inductor/kernel

TorchInductor also uses `fx.graph` to pattern-match and rewrite subgraphs:
🔗 https://github.com/pytorch/pytorch/tree/main/torch/_inductor/fx_passes

---

### 3. AOT Autograd
- Captures the **entire forward + backward graph** ahead of time.
- Allows cross-op optimization across training steps.
- Reduces runtime graph construction overhead.

---

## Why the compiled model can be slower (initially)

### 1. **Compilation overhead dominates early iterations**
- The first few executions include:
  - Graph capture (TorchDynamo)
  - Kernel generation (TorchInductor + Triton)
- On small batch sizes or short runs, this overhead can outweigh any runtime savings.

This is explicitly noted in the PyTorch documentation:
> *The first few iterations of a compiled model are expected to be slower due to compilation overhead.*

---

### 2. **Benchmarking artifacts**
Initial measurements were distorted by:
- Timing functions themselves being a bottleneck.
- CUDA event setup and synchronization overhead.
- Data generation and CPU↔GPU transfers inside the timed region.

After improving the benchmark (removing unnecessary JIT compilation of timing utilities and stabilizing warm-up behaviour), results became consistent and meaningful.

---

### 3. **Hardware limitations (important in our setup)**

Our system uses:

```

GPU: NVIDIA GeForce RTX 2050
Compute Capability: 8.x (consumer GPU)
Memory: 4GB

```

The official PyTorch tutorial explicitly states:

> *A modern NVIDIA GPU (V100, A100, or H100) is recommended to reproduce large speedups.*

On consumer GPUs:
- Less aggressive kernel fusion
- Lower memory bandwidth
- No large shared-memory advantages

➡️ As a result, **GPU speedups are smaller but still present**.

---

## Why `model.eval()` and `torch.no_grad()` are required

### `model.eval()`
Ensures:
- Dropout is disabled
- BatchNorm uses running statistics
- Deterministic inference behaviour

Without `eval()`:
- The graph may change between runs
- TorchDynamo may recompile
- Measurements become noisy and inconsistent

---

### `torch.no_grad()`
Ensures:
- No backward graph is constructed
- No gradient buffers are allocated
- AOT Autograd only captures forward execution

This is **critical** because:
- Inference benchmarking should measure **forward-only cost**
- Gradient tracking introduces extra ops and memory traffic
- Compiled training vs inference graphs are fundamentally different

---

## Comparing benchmarks

### Baseline benchmark (initial)
- Included timing overhead inside measured region
- Mixed warm-up and steady-state iterations
- Sometimes showed compiled model as slower

### Improved benchmark (final)
- Warm-up iterations performed implicitly
- Timing functions excluded from optimization path
- Stable measurements across runs

**Final consistent results:**

#### CPU
- Eager: **2.86 s**
- Compiled: **1.90 s**
- Speedup: **~1.5×**

#### GPU (RTX 2050)
- Eager: **0.171 s**
- Compiled: **0.133 s**
- Speedup: **~1.3×**

These results align with expectations for:
- Consumer GPUs
- Moderate batch sizes
- Models that are already fairly compute-dense

---

## Key takeaway

> `torch.compile` is not a magic speed switch. It trades **one-time compilation cost** for **lower steady-state runtime**, and the benefit depends on hardware, workload size, and correct benchmarking.

When measured correctly and run in steady state, `torch.compile` **does provide real speedups**, even on modest hardware — just smaller than what is seen on datacenter-class GPUs.

---

## References
- torch.compile overview  
  https://pytorch.org/docs/stable/compile.html
- TorchDynamo  
  https://github.com/pytorch/pytorch/tree/main/torch/_dynamo
- TorchInductor  
  https://github.com/pytorch/pytorch/tree/main/torch/_inductor
- FX passes in Inductor  
  https://github.com/pytorch/pytorch/tree/main/torch/_inductor/fx_passes
