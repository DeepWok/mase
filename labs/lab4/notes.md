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


## Q2 — Profiling the SDPA Kernel: Naive vs Fused Implementation

To understand the effect of kernel fusion in practice, we profiled both the naive Scaled Dot-Product Attention (SDPA) implementation and PyTorch’s fused SDPA kernel using the PyTorch profiler. Profiling was performed on both CPU and CUDA devices after warm-up, and outputs were verified to be numerically identical.

---

## Full Profiler Output

### CPU — Naive SDPA


```
    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  

    aten::bmm        97.30%        2.320s        98.37%        2.345s     117.248ms            20  
    aten::_softmax         1.00%      23.830ms         1.00%      23.830ms       2.383ms            10  
    aten::select         0.84%      20.097ms         1.04%      24.680ms       1.607us         15360  
    aten::mul         0.53%      12.640ms         0.54%      12.967ms       1.297ms            10  
```
Self CPU time total: 2.384s

---

### CPU — Fused SDPA

```
    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
    aten::_scaled_dot_product_flash_attention_for_cpu        99.07%      70.043ms        99.84%      70.584ms       7.058ms            10  
```


Self CPU time total: 70.697ms


---

### CUDA — Naive SDPA

```
    Name    Self CPU %      Self CPU   CPU total %     CPU total     Self CUDA    CUDA total  

    cudaDeviceSynchronize        80.35%       5.716ms        80.35%       5.716ms       0.000us       0.000us  
    aten::bmm        11.59%     824.566us        12.71%     904.437us       3.243ms       3.243ms  
    aten::_softmax         0.69%      49.247us         1.09%      77.402us       1.628ms       1.628ms  
```

Self CUDA time total: 6.486ms

---

### CUDA — Fused SDPA



```
    Name    Self CPU %      Self CPU   CPU total %     CPU total     Self CUDA    CUDA total  
    aten::_flash_attention_forward         8.32%     357.124us        47.12%       2.022ms       2.207ms       2.207ms  
    aten::_scaled_dot_product_flash_attention         4.50%     193.220us        56.97%       2.445ms       2.207ms       2.207ms  
```


Self CUDA time total: 2.207ms


---

## Summary Table

### Runtime Comparison

| Device | Implementation | Dominant Kernel | Total Runtime |
|------|----------------|-----------------|---------------|
| CPU  | Naive SDPA     | `aten::bmm`     | ~2.38 s       |
| CPU  | Fused SDPA     | FlashAttention (CPU) | ~70.7 ms |
| CUDA | Naive SDPA     | `aten::bmm` + softmax | ~6.49 ms |
| CUDA | Fused SDPA     | FlashAttention (CUDA) | ~2.21 ms |

---

## Discussion

On both CPU and CUDA, the fused SDPA implementation significantly reduces runtime compared to the naive implementation.

- **CPU:** The naive implementation is dominated by repeated large matrix multiplications and intermediate tensor materialization. Kernel fusion collapses these operations into a single optimized kernel, resulting in an approximate **33× speedup**.
- **CUDA:** While GPUs already handle matrix multiplications efficiently, the naive implementation still suffers from multiple kernel launches and synchronization overhead. The fused SDPA kernel reduces this overhead, yielding an approximate **3× speedup**.

Crucially, **the qualitative behaviour is the same on CPU and CUDA**:
- Naive SDPA executes attention as multiple independent operators.
- Fused SDPA executes attention as a single, optimized kernel.

This confirms that kernel fusion improves performance by reducing operator launch overhead and memory traffic, regardless of execution device, although the magnitude of the benefit depends on the hardware.

## Connection to FlashAttention

The fused SDPA kernel used by `torch.nn.functional.scaled_dot_product_attention` is an implementation of the **FlashAttention** algorithm. FlashAttention is not merely an optimized attention kernel; it is a **memory-efficient reformulation** of scaled dot-product attention that changes *how* the computation is executed rather than *what* is computed.

In the naive SDPA implementation, attention is decomposed into multiple high-level operations: matrix multiplication (`QKᵀ`), scaling, softmax, and a second matrix multiplication (`AV`). Each of these operations materializes intermediate tensors in global memory. On modern hardware, this leads to excessive memory reads and writes, making the computation **memory-bandwidth bound** rather than compute-bound.

FlashAttention addresses this by **fusing all attention steps into a single kernel** and computing attention **block-by-block**. Instead of storing the full attention matrix, it:
- Streams query, key, and value blocks from memory,
- Computes partial dot products on-chip,
- Applies a numerically stable softmax incrementally, and
- Accumulates the output without materializing intermediate tensors.

As a result, FlashAttention dramatically reduces memory traffic and synchronization overhead. This explains the profiler results observed in both CPU and CUDA runs:
- The naive SDPA path is dominated by `aten::bmm` and softmax kernels.
- The fused path collapses these operations into a single FlashAttention kernel, visible as `aten::_scaled_dot_product_flash_attention_*`.

The performance improvement is therefore a direct consequence of **kernel fusion and improved memory locality**, not just faster math. While the speedup is more pronounced on GPUs (where memory bandwidth is often the bottleneck), the same effect is observable on CPUs, confirming that FlashAttention’s benefits stem from reduced memory movement rather than device-specific tricks.

In summary, the fused SDPA kernel is a practical demonstration of FlashAttention’s core idea: *restructuring attention to be memory-efficient through kernel fusion*, which is why it consistently outperforms the naive implementation across devices.

### Relationship between `torch.compile` and FlashAttention

It is important to distinguish between the optimizations provided by `torch.compile` and those provided by FlashAttention.

`torch.compile` is a *general-purpose compilation framework* that captures PyTorch programs and applies graph-level optimizations such as operator fusion, kernel specialization, and code generation. These optimizations apply broadly across models and workloads, but they do not fundamentally change the underlying algorithm being executed.

FlashAttention, by contrast, is a *specialized algorithmic reformulation* of scaled dot-product attention. It changes how attention is computed by avoiding materialization of the attention matrix and performing the computation in a memory-efficient, tiled manner within a single kernel.

In practice, these two mechanisms are complementary:
- `torch.compile` can help reduce Python overhead and fuse compatible operators.
- FlashAttention provides a domain-specific kernel that addresses the memory-bandwidth bottleneck inherent in naive attention implementations.

The profiler results in this lab demonstrate that the largest gains for SDPA come from the use of a FlashAttention-style fused kernel, rather than from general compiler optimizations alone.

# Q3 — MXINT Quantization and Custom Kernels for Efficient Hardware Execution

### Motivation: Why Custom Kernels Matter for Quantization

While PyTorch provides highly optimized built-in kernels for common operations (e.g., `torch.matmul`, `torch.nn.functional.scaled_dot_product_attention`), these kernels are designed for general-purpose numerical formats such as FP32, FP16, or BF16. When models are quantized to custom formats, such as MXINT, the default kernels are no longer optimal.

To fully exploit the benefits of quantization, **custom kernels** are required. These kernels can be designed to:
- Operate directly on the quantized representation,
- Minimize unnecessary data conversion,
- Exploit hardware-friendly data layouts, and
- Reduce memory traffic and instruction overhead.

In this lab, a custom kernel is introduced for **MXINT dequantization**, demonstrating how numerical formats and kernel design interact to improve performance.

---

### Recap: MXINT Format

MXINT is a *block-scaled quantization format* that sits between floating-point and fixed-point representations. Instead of storing a full exponent per value (as in floating-point), MXINT groups several values together and shares a single exponent across the group.

An MXINT vector consists of:
- **One shared exponent** (8-bit, biased by 127), and
- **Multiple fixed-point mantissas**, each representing a scaled value.

This structure can be summarized as:

```

Exp |- Mantissa 1
|- Mantissa 2
|- ...
|- Mantissa group_size

```

The mantissas are signed fixed-point values, while the exponent provides a shared dynamic range. During dequantization, each mantissa is scaled by the same exponent factor.

---

### Why MXINT8 Benefits Custom Hardware

When **both weights and activations in a linear layer are quantized to MXINT8**, the format provides several advantages for custom hardware and specialized kernels:

#### 1. Reduced Memory Footprint
- MXINT8 significantly reduces the number of bits required per value compared to FP16 or FP32.
- Fewer bits mean lower memory bandwidth usage and better cache utilization.
- This directly improves performance on memory-bound workloads, such as large matrix multiplications.

#### 2. Simplified Arithmetic Units
- Mantissas are fixed-point values, which can be processed using **integer arithmetic** instead of full floating-point units.
- The exponent is shared across a group, so expensive per-element exponent handling is avoided.
- This enables simpler, smaller, and more energy-efficient compute units.

#### 3. Hardware-Friendly Dataflow
- The block structure of MXINT aligns naturally with SIMD, systolic arrays, and tensor-core-like architectures.
- A custom kernel can load one exponent and stream multiple mantissas through the same datapath.
- This increases arithmetic intensity and reduces instruction overhead.

#### 4. Reduced “Turing Tax”
Using MXINT8 enables hardware designers to move away from fully general-purpose execution toward **domain-specific accelerators**. By fixing the numerical format and dataflow:
- Instruction fetch and decode overhead is reduced,
- Control logic is simplified, and
- More transistors are dedicated to useful computation.

This directly reduces the *Turing Tax*—the performance and energy cost of using a universal programmable processor instead of specialized hardware.

---

### Role of the Custom Dequantization Kernel

The custom MXINT dequantization kernel shown in this lab illustrates how these benefits are realized in practice:

- The kernel loads a shared exponent once per group.
- Mantissas are converted from fixed-point to floating-point using simple scaling.
- Intermediate representations are minimized, reducing memory traffic.
- Dequantization is fused into the computation pipeline, avoiding unnecessary format conversions.

Compared to naïve dequantization approaches that operate element-wise with full floating-point logic, this kernel is both **faster** and **more memory-efficient**.

---

### Summary

MXINT8 benefits custom hardware when both weights and activations are quantized because it enables:
- Compact data representation,
- Integer-dominant arithmetic,
- Efficient block-wise computation, and
- Hardware designs that minimize general-purpose execution overhead.

In combination with custom kernels, MXINT8 allows accelerators to achieve high performance and energy efficiency while maintaining acceptable numerical accuracy. This mirrors the broader theme observed in this lab: **the largest performance gains arise not only from compiler optimizations, but from co-designing numerical formats, kernels, and hardware execution models.**

## Q3 — MXINT8 Dequantization Kernel: Dataflow, Thread Tiling, and Bit-Level Reconstruction

This section focuses on the **MXINT8 dequantization custom kernel** in `mase-cuda` and answers the embedded lab questions about:
1) the purpose of `dont_need_abs` / `dont_need_bias` and `bias`,  
2) how `cta_tiler` partitions global memory tensors using `local_tile`, and  
3) how `layout_sX` / `layout_tX` partition work across threads using `local_partition`.

---

### Overview: What the kernel does

Weights are stored in global memory in a quantized MXINT format:
- `x`: an `int8` mantissa per element (packed as raw int8 values)
- `scales`: a `uint8` exponent per group (shared micro-exponent)

During execution of a layer:
1. The GPU loads MXINT8 data from global memory.
2. The kernel **dequantizes** the mantissas to **BF16** using the shared exponent.
3. The dequantized BF16 values are written back to global memory for subsequent compute (e.g., GEMM).
4. After the layer finishes, the temporary BF16 weights can be discarded.

This saves GPU memory because the persistent storage is MXINT8 rather than BF16/FP16.

---

## Q3.1 — What is the purpose of `dont_need_abs` / `dont_need_bias` and `bias`?

In the host reference implementation (and mirrored in the device kernel), dequantization constructs a BF16 value by **bit-packing**:

```cpp
auto sign = static_cast<uint16_t>(hX[i] & 0x80) << 8;
auto exp  = static_cast<uint16_t>(hScales[i / group_size]) << 7;
auto mantissa_abs = abs(hX[i]);
auto frac = static_cast<uint16_t>((mantissa_abs & 0x3F) << 1);
auto out  = cutlass::bfloat16_t::bitcast(sign | exp | frac);
````

This produces a BF16 number whose:

* sign bit comes from MXINT mantissa sign,
* exponent comes from the shared scale,
* fraction bits come from the **lower 6 bits** of `|mantissa|`.

However, MXINT mantissas are not IEEE-normalized floats (no implicit leading 1).
To represent a wider signed fixed-point range efficiently, MXINT uses the `0x40` bit
(`0100_0000`) as a **region selector**.

```cpp
auto dont_need_abs = bool(mantissa_abs & 0x40);
auto bias = cutlass::bfloat16_t::bitcast(sign | exp | uint16_t(0));
y[i] = dont_need_abs ? out : out - bias;
```

**Interpretation:**

* If `mantissa_abs & 0x40` is **1**, the packed BF16 value `out` already represents the intended magnitude range → use `out` directly.
* If `mantissa_abs & 0x40` is **0**, the format expects the value to be **offset by one unit** in the exponent bucket → subtract `bias`.

`bias` is the BF16 value with:

* same sign and exponent,
* zero fraction,
  which acts like a baseline constant for that exponent bucket.

So `dont_need_abs` / `dont_need_bias` selects between two decoding rules:

* **Rule A:** `y = out`
* **Rule B:** `y = out - bias`

This is a common trick for encoding signed ranges with limited bits: one bit selects whether the decoded value is interpreted as a pure fraction or as a shifted/offset value.

---

## Q3.2 — How does `cta_tiler` partition the data for copy? (`local_tile`)

After reshaping, the kernel treats the mantissas as a 2D matrix:

```cpp
Tensor mX_raw = make_tensor(make_gmem_ptr(x_raw_int8), shape_x, stride_x);
Tensor mX = flatten(flat_divide(mX_raw, group_tiler)); 
// mX has shape (group_size, num_groups)
```

A CTA (threadblock) is assigned a tile:

```cpp
auto cta_tiler = make_shape(BLK_M, BLK_K);
auto cta_coord = make_coord(blockIdx.x, blockIdx.y);

Tensor gX = local_tile(mX, cta_tiler, cta_coord); // (BLK_M, BLK_K)
Tensor gY = local_tile(mY, cta_tiler, cta_coord); // (BLK_M, BLK_K)
```

This means:

* `blockIdx.x` selects which **row tile** (along `group_size`) the CTA owns.
* `blockIdx.y` selects which **column tile** (along `num_groups`) the CTA owns.

So the CTA loads/stores the block:

* rows: `[blockIdx.x * BLK_M, ..., blockIdx.x * BLK_M + BLK_M - 1]`
* cols: `[blockIdx.y * BLK_K, ..., blockIdx.y * BLK_K + BLK_K - 1]`

Scales are tiled only along the K dimension:

```cpp
Tensor gScale = local_tile(vScale, select<1>(cta_tiler), select<1>(cta_coord)); // (BLK_K,)
```

So each CTA loads the **BLK_K shared exponents** corresponding to the group columns it processes.

**In short:** `cta_tiler=(BLK_M,BLK_K)` divides the `(group_size × num_groups)` matrix into rectangular tiles, and `local_tile` selects the tile owned by `(blockIdx.x, blockIdx.y)`.

---

## Q3.3 — How does `layout_sX` partition threads for computation? (`local_partition`)

Threads are arranged using:

```cpp
auto layout_tX = make_layout(make_shape(thd_m, thd_k));
dim3 dimBlock(size(layout_tX)); // thd_m * thd_k threads
```

In each branch of the kernel, the code sets:

* `thd_m = BLK_M`
* `thd_k = BLK_K`

So the block has `BLK_M * BLK_K` threads — logically matching the tile area.

Thread-to-data mapping is created via:

```cpp
Tensor tXgX = local_partition(gX, layout_tX, threadIdx.x); // thread view of global tile
Tensor tXsX = local_partition(sX, layout_sX, threadIdx.x); // thread view of shared tile
```

With these layouts, each thread is assigned responsibility for the element(s) at its logical `(m,k)` position in the CTA tile.
This enables elementwise guarded copies:

```cpp
copy_if(tXpX, tXgX, tXsX);   // each thread copies its own element if in-bounds
...
copy_if(tXpX, tXrY, tXgY);   // each thread writes its own output if in-bounds
```

Predication is computed by partitioning an “identity tensor” with the same thread layout:

```cpp
Tensor cX = make_identity_tensor(shape(sX));               
Tensor tXcX = local_partition(cX, layout_sX, threadIdx.x); 
tXpX[i] = elem_less(tXcX[i], make_coord(m_max_coord, k_max_coord));
```

This ensures that edge CTAs (partial tiles) do not read/write out of bounds.

A useful sanity check is how scales are indexed:

```cpp
auto scaleIdx = threadIdx.x / size<0>(layout_tX);
auto exp = static_cast<uint16_t>(sScale[scaleIdx]) << 7;
```

Since `size<0>(layout_tX) == BLK_M`, dividing by `BLK_M` effectively groups threads by **K column**, so all `m` rows within the same column use the same shared exponent — matching the MXINT grouping semantics.
