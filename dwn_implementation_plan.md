# Differentiable Weightless Neural Networks (DWN) in MASE — Implementation Plan

**Branch:** `g4/dwn`
**Date:** 2026-02-18
**References:**
- [DWN Paper (ICML 2024)](https://arxiv.org/abs/2410.11112)
- [DWN GitHub](https://github.com/alanbacellar/DWN)
- [DiffLogicNet Paper](https://arxiv.org/abs/2210.08277)
- [DiffLogicNet Repo](https://github.com/Felix-Petersen/difflogic)
- [Martha's Paper – NeuraLUT-Assemble (2025)](https://arxiv.org/abs/2504.00592)
- [Existing MASE DiffLogic PR #276](https://github.com/DeepWok/mase/pull/276)

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [DWN Technical Architecture](#2-dwn-technical-architecture)
3. [Comparison: DWN vs DiffLogicNet](#3-comparison-dwn-vs-difflogicnet)
4. [MASE Integration Architecture](#4-mase-integration-architecture)
5. [Phase 1 – Python Modules (Inference Components)](#phase-1--python-modules-inference-components)
6. [Phase 2 – MaseGraph Transform Pass](#phase-2--masegraph-transform-pass)
7. [Phase 3 – Verilog Emit Pass](#phase-3--verilog-emit-pass)
8. [Phase 4 – Benchmarking](#phase-4--benchmarking)
9. [Parameter Sweep Dimensions](#parameter-sweep-dimensions)
10. [File Structure](#file-structure)
11. [Testing Strategy](#testing-strategy)
12. [Extension Targets](#extension-targets)

---

## 1. Background and Motivation

### The Problem with DiffLogicNet

DiffLogicNet (ICML 2022) makes boolean logic gates differentiable by assigning each gate a real-valued weight vector over 16 possible 2-input logic functions. The number of parameters per LUT scales as **2^(2^n)**:

| LUT Inputs (n) | Parameters per LUT | Comment |
|---|---|---|
| 2 | 16 | DiffLogicNet supports this |
| 4 | 65,536 | Impractical |
| 6 | 18,446,744,073,709,551,616 | Impossible |

Modern Xilinx FPGAs use **LUT-6** (6-input lookup tables) as their fundamental logic primitive. DiffLogicNet's double-exponential parameterisation makes it impossible to target LUT-6s, leaving the majority of FPGA fabric unutilised.

### The DWN Solution

Differentiable Weightless Neural Networks (ICML 2024) solve this by **directly learning the lookup table contents** rather than parameterising the logic function space. Parameters scale as **2^n** (linear in the exponent):

| LUT Inputs (n) | Parameters per LUT |
|---|---|
| 2 | 4 |
| 4 | 16 |
| 6 | 64 |

Differentiability is recovered through the **Extended Finite Difference (EFD)** technique, which approximates gradients through the discrete LUT lookup.

**Reported improvements over DiffLogicNet (ICML 2024 paper):**
- Up to **135× energy reduction** on Xilinx FPGAs
- Up to **42.8× circuit area reduction** on Xilinx FPGAs
- Superior or equal accuracy on standard classification benchmarks
- Efficient LUT-6 utilisation, unlocking modern Xilinx Artix-7 / UltraScale+ devices

---

## 2. DWN Technical Architecture

### 2.1 Inference Pipeline

```
Raw Input (float/fixed-point)
        │
        ▼
┌─────────────────────────────┐
│  DistributiveThermometer    │
│  Encoding                   │
│  in:  [B, F]                │
│  out: [B, F × T]  (binary)  │
└─────────────────────────────┘
        │
        ▼  (flatten to binary vector)
┌─────────────────────────────┐
│  LUTLayer × L               │  (L stacked layers)
│  in:  [B, W_in]  (binary)   │
│  out: [B, W_out] (binary)   │
│  params: W_out × 2^n        │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  GroupSum                   │
│  in:  [B, W_out] (binary)   │
│  out: [B, C]     (integer)  │
│  → popcount per class group │
└─────────────────────────────┘
        │
        ▼
  Class Logits / Softmax
```

### 2.2 DistributiveThermometer Encoding

Converts continuous-valued inputs to binary vectors using learned per-feature thresholds.

**API (from `torch_dwn`):**
```python
enc = DistributiveThermometer(num_bits=8, feature_wise=True)
enc.fit(X_train)            # learn per-feature thresholds
x_binary = enc.binarize(X) # shape: (*X.shape, num_bits)
```

**Parameters to sweep:**
- `num_bits` (T): number of thresholds per feature. More bits → finer resolution but wider binary input to LUTLayer.
- Input data width (for fixed-point hardware path): number of bits in the upstream fixed-point representation.

**Hardware representation:**
Each threshold is a fixed-point comparator. T comparators per feature, output is a T-bit thermometer code.

### 2.3 LUTLayer

The core trainable component. Each output neuron corresponds to one LUT with n inputs drawn (via a mapping) from the binary input vector.

**API:**
```python
LUTLayer(
    input_size,          # width of binary input vector
    output_size,         # number of LUTs (= width of binary output)
    n,                   # LUT fan-in (typically 2, 4, or 6 for FPGA)
    mapping='random',    # 'random' | 'learnable' | 'arange'
    alpha=None,
    beta=None,
    ste=True,            # Straight-Through Estimator for binarisation
    clamp_luts=True,
    lm_tau=0.001,        # temperature for learnable mapping
)
```

**Parameters:**
- `output_size × 2^n` real-valued weights clamped to [−1, 1]
- During inference: weights are binarised; LUT address is formed from the n selected binary inputs

**Gradient flow (EFD):**
The Extended Finite Difference approximation treats each LUT as a differentiable function during the backward pass, enabling training with standard gradient descent.

**Key design notes (from DWN paper and repo):**
- Use `mapping='learnable'` only in the **first layer** — this substantially improves accuracy with minimal overhead.
- `mapping='random'` for all subsequent layers is sufficient and faster to converge.
- `ste=True` (Straight-Through Estimator) is the recommended default for binarisation.

### 2.4 GroupSum

Aggregates the binary output of the last LUTLayer into integer class scores via a grouped population count (popcount).

**API:**
```python
GroupSum(
    num_groups,   # number of output classes (C)
    tau=1,        # softmax temperature (crucial hyperparameter)
)
```

**Operation:** Splits the binary output vector of width `W` into `C` equal groups of size `W/C`. Applies popcount to each group → integer score per class. Optionally applies temperature scaling before softmax.

**Hardware:** Popcount is a native FPGA primitive (`$countones` in SystemVerilog). No multipliers required.

### 2.5 Learnable Mapping and Spectral Regularisation

**Learnable Mapping:**
In the first LUTLayer, instead of fixed random input connections, the mapping itself is learnable via a soft assignment matrix with Gumbel-softmax relaxation. This allows the network to discover optimal feature groupings.

**Spectral Regularisation:**
Encourages LUT weight matrices to have low spectral norm, penalising overly complex lookup functions. Applied as an auxiliary loss term during training:
```
L_total = L_CE + λ × Σ_l σ_max(W_l)
```
where σ_max is the largest singular value of the LUT weight matrix for layer l.

---

## 3. Comparison: DWN vs DiffLogicNet

| Property | DiffLogicNet | DWN |
|---|---|---|
| **Parameterisation** | 2^(2^n) per gate | 2^n per LUT |
| **LUT-2 params** | 16 | 4 |
| **LUT-4 params** | 65,536 | 16 |
| **LUT-6 params** | ~1.8 × 10^19 | 64 |
| **FPGA LUT size** | LUT-2 only | LUT-2, 4, 6 (native) |
| **Gradient mechanism** | Continuous relaxation of gate probs | Extended Finite Difference |
| **Training stability** | Requires grad_factor for deep nets | STE + EFD stable |
| **Hardware efficiency** | C-code only (no FPGA LUT-6 support) | Native Xilinx LUT-6 support |
| **Energy (vs DiffLogicNet)** | baseline | up to 135× lower |
| **Area (vs DiffLogicNet)** | baseline | up to 42.8× lower |

**NeuraLUT-Assemble (Martha's paper, 2025)** provides additional context:
- Uses mixed-precision and assembly of larger neurons from smaller units
- Achieves up to **8.42× reduction in area-delay product** vs prior SOTA
- Evaluated on network intrusion detection, MNIST, jet classification
- DWN should be benchmarked in this same evaluation space

---

## 4. MASE Integration Architecture

DWN will follow the same three-layer integration pattern as LUTNet and LogicNets in MASE:

```
┌─────────────────────────────────────────────────────┐
│              User Configuration (TOML)              │
│         configs/dwn/dwn_lut{2,4,6}.toml            │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│          MaseGraph Transform Pass                   │
│   src/chop/passes/graph/transforms/dwn/             │
│   dwn_transform_pass(graph, pass_args)              │
│   → iterates graph nodes, replaces nn.Linear with  │
│     DWN modules; inserts Thermometer + GroupSum     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           DWN PyTorch Modules                       │
│   src/chop/nn/dwn/                                  │
│   ├── thermometer.py   (DistributiveThermometer)    │
│   ├── lut_layer.py     (LUTLayer)                  │
│   └── group_sum.py     (GroupSum)                  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           Verilog Emit Pass                         │
│   src/mase_components/dwn_layers/                   │
│   ├── rtl/fixed_dwn_thermometer.sv                 │
│   ├── rtl/fixed_dwn_lut_layer.sv                   │
│   ├── rtl/fixed_dwn_lut_neuron.sv                  │
│   └── rtl/fixed_dwn_groupsum.sv                    │
└─────────────────────────────────────────────────────┘
```

---

## Phase 1 – Python Modules (Inference Components)

**Goal:** Implement standalone, differentiable DWN components in PyTorch that can be used independently of MASE's graph infrastructure.

### 1.1 Directory Structure

```
src/chop/nn/dwn/
├── __init__.py
├── thermometer.py      # DistributiveThermometer
├── lut_layer.py        # LUTLayer with EFD
└── group_sum.py        # GroupSum
```

### 1.2 `DistributiveThermometer` (`thermometer.py`)

```python
class DistributiveThermometer(nn.Module):
    """
    Encodes a fixed-point input vector as a binary thermometer code.

    Args:
        num_bits (int):      Number of thresholds per feature (T)
        feature_wise (bool): Learn separate thresholds per feature

    Shape:
        Input:  (*, F)         float / fixed-point
        Output: (*, F * T)     binary {0, 1}
    """
    def __init__(self, num_bits: int, feature_wise: bool = True): ...

    def fit(self, x: torch.Tensor) -> None:
        """Learn thresholds from training data statistics."""
        ...

    def binarize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply thermometer encoding."""
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.binarize(x)
```

**Hardware parameters exposed:**
- `num_bits`: directly maps to number of comparators in hardware
- `input_width`: fixed-point width of upstream data (for Verilog emit)
- `input_frac_width`: fixed-point fractional width

### 1.3 `LUTLayer` (`lut_layer.py`)

```python
class LUTLayer(nn.Module):
    """
    A layer of interconnected lookup tables with differentiable training.

    Args:
        input_size (int):    Width of binary input vector
        output_size (int):   Number of LUTs (= width of binary output)
        n (int):             LUT fan-in / number of inputs per LUT (2, 4, or 6)
        mapping (str):       'random' | 'learnable' | 'arange'
        alpha (float|None):  EFD finite difference step size (None = auto)
        beta (float|None):   EFD decay factor (None = auto)
        ste (bool):          Use Straight-Through Estimator for binarisation
        clamp_luts (bool):   Clamp LUT weights to [-1, 1]
        lm_tau (float):      Learnable mapping temperature

    Parameters:
        lut_weights:  (output_size, 2^n)  – the lookup table contents
        mapping_idx:  (output_size, n)    – which inputs each LUT reads

    Shape:
        Input:  (*, input_size)    binary
        Output: (*, output_size)   binary

    Note on alpha/beta: These are EFD hyperparameters from the torch_dwn reference
    implementation. alpha controls the finite difference step size; beta controls
    exponential decay of the step size over training. Both default to None (auto-
    computed internally). Expose them for hyperparameter sweeps but leave None
    for the default training configuration.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n: int = 6,
        mapping: str = 'random',
        alpha: float = None,
        beta: float = None,
        ste: bool = True,
        clamp_luts: bool = True,
        lm_tau: float = 0.001,
    ): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Gather n inputs per LUT using mapping_idx
        # 2. Form 2^n-dimensional one-hot address vector
        # 3. Dot product with lut_weights → raw output
        # 4. Binarise output (STE in backward)
        ...

    def get_lut_contents(self) -> torch.Tensor:
        """Return binarised LUT contents for hardware export. Shape: (output_size, 2^n)"""
        ...
```

**Spectral regularisation loss:**
```python
def spectral_reg_loss(layer: LUTLayer, lambda_reg: float = 1e-4) -> torch.Tensor:
    """Compute spectral regularisation penalty for a LUTLayer."""
    W = layer.lut_weights  # (output_size, 2^n)
    sigma_max = torch.linalg.matrix_norm(W, ord=2)
    return lambda_reg * sigma_max
```

### 1.4 `GroupSum` (`group_sum.py`)

```python
class GroupSum(nn.Module):
    """
    Grouped population count aggregation for classification.

    Splits binary input into C equal groups, applies popcount per group,
    then applies temperature-scaled softmax.

    Args:
        num_groups (int):  Number of output classes (C)
        tau (float):       Softmax temperature (crucial hyperparameter)

    Shape:
        Input:  (B, W)  binary, W must be divisible by num_groups
        Output: (B, C)  integer counts (before softmax)
    """
    def __init__(self, num_groups: int, tau: float = 1.0): ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to (B, C, W//C) then sum along last dim
        ...
```

### 1.5 DWN Model Class

```python
class DWNModel(nn.Module):
    """
    End-to-end DWN model: Thermometer → LUTLayer × L → GroupSum

    Args:
        input_features (int):   Number of input features (F)
        num_classes (int):      Number of output classes (C)
        num_bits (int):         Thermometer bits per feature (T)
        hidden_size (int):      LUT layer width (W)
        num_layers (int):       Number of stacked LUT layers (L)
        lut_n (int):            LUT fan-in (2, 4, or 6)
        mapping_first (str):    Mapping strategy for first LUT layer
        mapping_rest (str):     Mapping strategy for subsequent layers
        tau (float):            GroupSum temperature
        lambda_reg (float):     Spectral regularisation weight
    """
```

---

## Phase 2 – MaseGraph Transform Pass

**Goal:** Create a MASE transform pass that converts a standard neural network (typically an `nn.Linear`-based model like LFC) into a DWN model by replacing linear layers and inserting thermometer encoding and GroupSum.

### 2.1 Pass Structure

```
src/chop/passes/graph/transforms/dwn/
├── __init__.py         # exports dwn_transform_pass, dwn_fusion_transform_pass
├── quantize.py         # main graph iterator and module replacement
├── fusion.py           # fuses BatchNorm etc. after conversion
└── config_parser.py    # parses DWN config from TOML
```

### 2.2 Main Transform Pass

```python
# src/chop/passes/graph/transforms/dwn/quantize.py

DWN_CONVERTIBLE_OPS = ("linear",)  # extend to conv1d/conv2d later

def dwn_transform_pass(graph: MaseGraph, pass_args: dict = None) -> tuple:
    """
    Convert a standard neural network to a DWN.

    pass_args example:
        {
            "by": "type",
            "default": {"config": {"name": None}},
            "linear": {
                "config": {
                    "name": "dwn",
                    "lut_n": 6,              # LUT fan-in
                    "hidden_size": 2000,     # LUT layer width
                    "num_layers": 2,         # stacked LUT layers
                    "mapping_first": "learnable",
                    "mapping_rest": "random",
                    "num_bits": 8,           # thermometer bits
                    "tau": 3.33,             # GroupSum temperature
                    "lambda_reg": 1e-4,      # spectral reg weight
                }
            },
        }
    """
    by = pass_args.pop("by")
    match by:
        case "type":
            graph = _graph_iterator_dwn_by_type(graph, pass_args)
        case "name":
            graph = _graph_iterator_dwn_by_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported DWN "by": {by}')

    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}
```

### 2.3 Module Registration

**Important:** DWN modules are **not** drop-in quantized variants with the `(in_features, out_features, bias, config)` constructor signature that `modify.py:create_new_module` expects. Therefore, DWN must **not** register in `quantized_module_map`. Instead, write a standalone graph iterator in `dwn/quantize.py` that bypasses `modify.py` entirely:

```python
# In src/chop/passes/graph/transforms/dwn/quantize.py
from chop.nn.dwn import LUTLayer, DistributiveThermometer, GroupSum

def _graph_iterator_dwn_by_type(graph, config: dict):
    """
    Standalone iterator — does NOT call create_new_module() or quantized_module_map.
    Directly constructs DWN modules from config and replaces nn.Linear nodes.
    """
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in DWN_CONVERTIBLE_OPS:
            continue
        node_config = config.get(get_mase_op(node), config.get("default", {})).get("config", {})
        if node_config.get("name") != "dwn":
            continue
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            new_module = LUTLayer(
                input_size=ori_module.in_features,
                output_size=node_config["hidden_size"],
                n=node_config["lut_n"],
                mapping=node_config.get("mapping_first", "learnable"),
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph
```

### 2.4 Pass Registration

Two explicit steps are required — both must be done:

**Step 1** — Add import to `src/chop/passes/graph/transforms/__init__.py`:

```python
# transforms/__init__.py  (append to existing imports)
from .dwn import dwn_transform_pass, dwn_fusion_transform_pass
```

**Step 2** — Add entry to the `PASSES` dict in `src/chop/passes/graph/__init__.py`:

```python
# passes/graph/__init__.py — add to PASSES dict (around line 90)
PASSES = {
    ...
    "quantize": quantize_transform_pass,
    "dwn": dwn_transform_pass,           # <-- add this
    "dwn_fusion": dwn_fusion_transform_pass,  # <-- and this
    ...
}
```

Without Step 2, the pass is importable but not invocable via `graph.run_passes(["dwn", ...])`.

> **Note:** The difflogic transform pass (from PR #276) was removed in PR #307. The DWN pass re-introduces this category. Check `PASSES` in `passes/graph/__init__.py` at the time of implementation to ensure no key conflict.

### 2.5 Configuration Files

```toml
# configs/dwn/dwn_lut6.toml
[passes.dwn]
by = "type"

[passes.dwn.default.config]
name = "NA"

[passes.dwn.linear.config]
name       = "dwn"
lut_n      = 6
hidden     = 2000
num_layers = 2
num_bits   = 8
mapping    = "learnable"   # first layer; rest use "random"
tau        = 3.33
lambda_reg = 1e-4
```

---

## Phase 3 – Verilog Emit Pass

**Goal:** Generate synthesisable SystemVerilog for Xilinx FPGAs from a trained DWN model, targeting native LUT-6 primitives.

> **Note:** The hardware/Verilog infrastructure was removed in PR #307. DWN will re-introduce it specifically for the DWN component set, following the pattern established by the difflogic Verilog emit (visible in git history as commit `ac5d357f`).

### 3.1 Hardware Directory Structure

```
src/mase_components/dwn_layers/
├── __init__.py
├── passes.py                          # MaseGraph hardware metadata passes
├── rtl/
│   ├── fixed_dwn_flatten.sv           # Reshape float→binary boundary (mirrors fixed_difflogic_flatten.sv)
│   ├── fixed_dwn_thermometer.sv       # Thermometer encoding comparators
│   ├── fixed_dwn_lut_neuron.sv        # Single LUT-n unit (purely combinational)
│   ├── fixed_dwn_lut_layer.sv         # Parameterised array of LUT neurons + valid/ready
│   └── fixed_dwn_groupsum.sv          # Grouped popcount aggregator + valid/ready
└── test/
    ├── fixed_dwn_flatten_tb.py
    ├── fixed_dwn_thermometer_tb.py
    ├── fixed_dwn_lut_layer_tb.py
    └── fixed_dwn_groupsum_tb.py
```

**Note on `fixed_dwn_flatten.sv`:** This module handles the boundary between the fixed-point input domain and the binary thermometer domain. It mirrors `fixed_difflogic_flatten.sv` from PR #276 and is required by the hardware metadata pass (`dwn_hardware_force_fixed_flatten_pass`). Without it, the MASE Verilog emit cannot correctly wire the input datapath.

### 3.2 RTL Module Descriptions

#### `fixed_dwn_thermometer.sv`

```
Parameters:
  DATA_IN_0_TENSOR_SIZE_DIM_0   - number of input features (F)
  DATA_IN_0_WIDTH               - fixed-point input width (bits)
  DATA_IN_0_FRAC_WIDTH          - fractional bits
  NUM_THRESHOLDS                - thermometer bits per feature (T)

Ports:
  data_in_0   [F×DATA_IN_0_WIDTH-1:0]  - packed fixed-point input
  data_out_0  [F×T-1:0]                - packed binary thermometer output

Logic:
  For each feature f and threshold t:
    data_out_0[f*T+t] = (data_in_0[f] >= threshold[f][t]) ? 1 : 0
  Thresholds stored as ROM parameters (loaded from trained model).
```

#### `fixed_dwn_lut_neuron.sv`

Purely combinational — no clk/rst/valid/ready needed at this level.

```
Parameters:
  LUT_N                    - fan-in (2, 4, or 6)
  LUT_CONTENTS [2^N-1:0]  - the learned binary LUT contents (ROM)

Ports:
  data_in_0  [LUT_N-1:0]  - selected binary inputs (combinational)
  data_out_0              - single bit output (combinational)

Logic:
  data_out_0 = LUT_CONTENTS[data_in_0]  // direct LUT address, maps to 1 FPGA LUT-n
```

#### `fixed_dwn_lut_layer.sv`

Follows the MASE valid/ready handshake protocol (same as `fixed_difflogic_logic.sv`).

```
Parameters:
  DATA_IN_0_TENSOR_SIZE_DIM_0   - input binary vector width
  DATA_OUT_0_TENSOR_SIZE_DIM_0  - output binary vector width (= num LUTs)
  LUT_N                         - fan-in per LUT
  INPUT_INDICES [OUT×N-1:0]     - which inputs each LUT reads (packed)
  LUT_CONTENTS  [OUT×2^N-1:0]  - all LUT contents (packed)
  PIPELINE_STAGES               - 0: combinational; 1+: register stages

Ports:
  clk, rst
  data_in_0        [(DATA_IN_0_TENSOR_SIZE_DIM_0-1):0]
  data_in_0_valid  logic
  data_in_0_ready  logic   (output)
  data_out_0       [(DATA_OUT_0_TENSOR_SIZE_DIM_0-1):0]
  data_out_0_valid logic   (output)
  data_out_0_ready logic

Logic:
  Generate DATA_OUT_0_TENSOR_SIZE_DIM_0 instances of fixed_dwn_lut_neuron,
  each reading LUT_N inputs selected by INPUT_INDICES.
  Pass-through handshake when PIPELINE_STAGES=0:
    assign data_out_0_valid = data_in_0_valid;
    assign data_in_0_ready  = data_out_0_ready;
  With PIPELINE_STAGES=1: insert one always_ff register stage.
```

#### `fixed_dwn_groupsum.sv`

Directly extends `fixed_difflogic_groupsum.sv`. Output width is **derived** (not a parameter) to match the difflogic pattern and avoid emit-pass miscalculation:

```
Parameters:
  DATA_IN_0_TENSOR_SIZE_DIM_0   - binary input width (W)
  DATA_OUT_0_TENSOR_SIZE_DIM_0  - number of classes (C)
  USE_PIPELINED_ADDER_TREE      - 0: $countones (default), 1: adder tree

Ports:
  clk, rst
  data_in_0        [DATA_IN_0_TENSOR_SIZE_DIM_0-1:0]
  data_in_0_valid  logic
  data_in_0_ready  logic   (output)
  data_out_0       [$clog2(DATA_IN_0_TENSOR_SIZE_DIM_0/DATA_OUT_0_TENSOR_SIZE_DIM_0):0]
                   [0:DATA_OUT_0_TENSOR_SIZE_DIM_0-1]  (array)
  data_out_0_valid logic   (output)
  data_out_0_ready logic

Logic:
  localparam GROUP_SIZE = DATA_IN_0_TENSOR_SIZE_DIM_0 / DATA_OUT_0_TENSOR_SIZE_DIM_0;
  For each class c:
    data_out_0[c] = $countones(data_in_0[c*GROUP_SIZE +: GROUP_SIZE])
  Output width = $clog2(GROUP_SIZE)+1 bits (auto-derived, not parameterised)
  Pass-through handshake (combinational path):
    assign data_out_0_valid = data_in_0_valid;
    assign data_in_0_ready  = data_out_0_ready;
```

### 3.3 Verilog Emit Pass

Three hardware metadata passes are required, mirroring the difflogic pattern from PR #276:

```python
# src/mase_components/dwn_layers/passes.py

def dwn_hardware_metadata_pass(graph, args={}):
    """
    Add hardware metadata to DWN LUTLayer nodes for Verilog emit.
    Modelled after difflogic_hardware_metadata_optimize_pass.
    """
    for node in graph.nodes:
        if _is_dwn_lut_node(node):
            pre_common_args_md = node.meta["mase"]["common"]["args"]
            post_common_args_md = {}
            node.meta["mase"]["hardware"]["dwn_args"] = {}
            for k, v in pre_common_args_md.items():
                if "data_in" not in k:
                    node.meta["mase"]["hardware"]["dwn_args"][k] = v
                else:
                    post_common_args_md[k] = v
            node.meta["mase"]["common"]["args"] = OrderedDict(post_common_args_md)
            node.meta["mase"]["hardware"]["toolchain"] = "INTERNAL_RTL"
            node.meta["mase"]["hardware"]["module"] = "fixed_dwn_lut_layer"
            node.meta["mase"]["hardware"]["dependence_files"] = [
                "dwn_layers/rtl/fixed_dwn_lut_neuron.sv",
                "dwn_layers/rtl/fixed_dwn_lut_layer.sv",
            ]
            # Embed binarised LUT contents + input indices as Verilog parameters
            # DATA_OUT_0_WIDTH is NOT set here — it is auto-derived in RTL via $clog2
            node.meta["mase"]["hardware"]["dwn_args"].update({
                "lut_n": node.meta["mase"]["common"]["args"].get("lut_n", 6),
                "lut_contents": _extract_lut_contents(node),   # shape: (output_size, 2^n) binary
                "input_indices": _extract_input_indices(node),  # shape: (output_size, n) int
            })
    return graph, None


def dwn_hardware_force_fixed_flatten_pass(graph, args={}):
    """
    Force the flatten node (float→binary boundary) to use fixed_dwn_flatten RTL.
    Mirrors difflogic_hardware_force_fixed_flatten_pass from PR #276.
    """
    for node in graph.nodes:
        if node.meta["mase"]["common"].get("mase_op") == "flatten":
            node.meta["mase"]["hardware"]["toolchain"] = "INTERNAL_RTL"
            node.meta["mase"]["hardware"]["module"] = "fixed_dwn_flatten"
            node.meta["mase"]["hardware"]["dependence_files"] = [
                "dwn_layers/rtl/fixed_dwn_flatten.sv"
            ]
            add_verilog_param(node)
            add_extra_verilog_param(node, graph)
            graph.meta["mase"]["hardware"]["verilog_sources"] += \
                node.meta["mase"]["hardware"]["dependence_files"]
    return graph, None


def dwn_hardware_groupsum_pass(graph, args={}):
    """Add hardware metadata to GroupSum nodes."""
    for node in graph.nodes:
        if _is_dwn_groupsum_node(node):
            node.meta["mase"]["hardware"]["toolchain"] = "INTERNAL_RTL"
            node.meta["mase"]["hardware"]["module"] = "fixed_dwn_groupsum"
            node.meta["mase"]["hardware"]["dependence_files"] = [
                "dwn_layers/rtl/fixed_dwn_groupsum.sv"
            ]
            # DATA_OUT width is auto-derived in RTL via $clog2 — do NOT set explicitly
    return graph, None
```

### 3.4 Weight Export for Hardware

After training, extract the binarised LUT contents and input indices:

```python
def export_dwn_to_verilog_params(model: DWNModel) -> dict:
    """
    Extract binarised LUT contents and connectivity from a trained DWN model.

    Returns:
        dict with keys:
            'thermometer_thresholds': np.ndarray  shape (F, T)
            'lut_layers': list of dicts, each with:
                'lut_contents':   np.ndarray shape (output_size, 2^n)  binary
                'input_indices':  np.ndarray shape (output_size, n)    integer
            'num_classes': int
            'group_size': int
    """
```

---

## Phase 4 – Benchmarking

**Goal:** Quantitatively compare DWN, DiffLogicNet (MASE PR #276), and NeuraLUT-Assemble on the metrics that matter for FPGA deployment.

### 4.1 Metrics

| Metric | Description | Tool |
|---|---|---|
| **Model accuracy** | Top-1 classification accuracy on test set | PyTorch |
| **Parameter count** | Total trainable parameters | `sum(p.numel() for p in model.parameters())` |
| **LUT utilisation** | FPGA LUTs consumed per equivalent network width | Vivado synthesis report |
| **Latency (cycles)** | Clock cycles from first input to last output | RTL simulation (cocotb) |
| **Area** | Post-synthesis LUT + FF count | Vivado synthesis |
| **Area-delay product** | Area × Latency (primary FPGA efficiency metric) | Derived |
| **Energy** | Post-implementation power estimate (mW) × latency (ms) | Vivado power analysis |
| **Training time** | Wall-clock time to convergence | `time` module |

### 4.2 Datasets

- **MNIST** (primary — standard DWN/DiffLogicNet benchmark, rapid iteration)
- **NSL-KDD** (network intrusion detection — tabular, used in NeuraLUT-Assemble paper)
- **Jet classification** (HEP tabular dataset — used in NeuraLUT-Assemble paper)

> **Note:** CIFAR-10 is **not** appropriate for DWN. DWN operates on flattened, thermometer-encoded features and has no spatial convolution. Feeding CIFAR-10 as a 3072-feature flat vector would produce an extremely wide binary input and would not constitute a meaningful comparison against NeuraLUT-Assemble or DiffLogicNet (neither of which targets raw CIFAR-10 either). Stick to the evaluation space established by both papers.

### 4.3 Benchmark Script Structure

```
test/benchmarks/dwn_vs_difflogic/
├── train_dwn.py           # Train DWN with configurable hyperparams
├── train_difflogic.py     # Train DiffLogicNet (from MASE PR #276)
├── eval_accuracy.py       # Evaluate accuracy on test set
├── run_synthesis.py       # Run Vivado synthesis and collect resource reports
├── plot_results.py        # Generate comparison plots
└── configs/
    ├── dwn_lut2.toml
    ├── dwn_lut4.toml
    ├── dwn_lut6.toml
    └── difflogic.toml
```

### 4.4 Parameter Count Comparison

For equivalent network capacity (same number of FPGA LUTs used):

| Scheme | LUTs | Params per LUT | Total params (1000 LUTs) |
|---|---|---|---|
| DiffLogicNet (LUT-2) | 1000 × 1 LUT-2 | 16 | 16,000 |
| DWN (LUT-2) | 1000 × 1 LUT-2 | 4 | 4,000 |
| DWN (LUT-4) | 250 × 1 LUT-4 | 16 | 4,000 |
| DWN (LUT-6) | 63 × 1 LUT-6 | 64 | 4,032 |

DWN with LUT-6 achieves **16× fewer software parameters** than DiffLogicNet LUT-2 for the same FPGA resource, enabling much better utilisation of Xilinx fabric.

### 4.5 Expected Results (from DWN paper)

- DWN LUT-6 ≥ DiffLogicNet accuracy on standard benchmarks
- DWN area-delay product: ~42.8× lower than DiffLogicNet
- DWN energy: ~135× lower than DiffLogicNet

---

## Parameter Sweep Dimensions

### Thermometer Encoding

| Parameter | Values to Sweep | Effect |
|---|---|---|
| `num_bits` (T) | 2, 4, 8, 16 | Precision vs. input width trade-off |
| `feature_wise` | True, False | Per-feature vs. global thresholds |
| `input_data_width` | 4, 8, 16 | Fixed-point resolution (hardware path) |

### LUT Layer

| Parameter | Values to Sweep | Effect |
|---|---|---|
| `lut_n` | 2, 4, 6 | FPGA LUT size; key hardware parameter |
| `hidden_size` (W) | 500, 1000, 2000, 4000 | Model width / FPGA area |
| `num_layers` (L) | 1, 2, 3, 4 | Model depth |
| `mapping_first` | 'random', 'learnable' | Accuracy vs. convergence speed |
| `pipeline_stages` | 0, 1 | [Extension] Timing closure |

### GroupSum

| Parameter | Values to Sweep | Effect |
|---|---|---|
| `tau` | 0.1, 0.3, 1.0, 3.33 | Softmax sharpness; critical for accuracy |
| `output_precision` | ceil(log2(W/C))+1 bits | Integer width in hardware |
| `adder_tree` | False (countones), True | [Extension] Timing vs. area |

### Training

| Parameter | Default | Notes |
|---|---|---|
| Optimizer | Adam | As per DWN paper |
| Learning rate | 0.01 | Higher than standard (0.001) |
| LR Schedule | StepLR, γ=0.1, step=14 | From DWN MNIST example |
| Batch size | 32 | Standard |
| Epochs | 30 | Standard |
| `lambda_reg` | 1e-4 | Spectral regularisation weight |

---

## File Structure

```
mase/
├── src/chop/
│   ├── nn/
│   │   └── dwn/
│   │       ├── __init__.py
│   │       ├── thermometer.py          # DistributiveThermometer
│   │       ├── lut_layer.py            # LUTLayer + EFD gradient
│   │       └── group_sum.py            # GroupSum
│   └── passes/graph/
│       └── transforms/
│           └── dwn/
│               ├── __init__.py         # exports transform passes
│               ├── quantize.py         # dwn_transform_pass
│               ├── fusion.py           # dwn_fusion_transform_pass
│               └── config_parser.py    # TOML config parser
│
├── src/mase_components/
│   └── dwn_layers/
│       ├── __init__.py
│       ├── passes.py                   # Hardware metadata passes
│       ├── rtl/
│       │   ├── fixed_dwn_thermometer.sv
│       │   ├── fixed_dwn_lut_neuron.sv
│       │   ├── fixed_dwn_lut_layer.sv
│       │   └── fixed_dwn_groupsum.sv
│       └── test/
│           ├── Makefile
│           ├── fixed_dwn_thermometer_tb.py
│           ├── fixed_dwn_lut_layer_tb.py
│           └── fixed_dwn_groupsum_tb.py
│
├── configs/dwn/
│   ├── dwn_lut2.toml
│   ├── dwn_lut4.toml
│   └── dwn_lut6.toml
│
├── test/
│   └── passes/graph/transforms/
│       └── dwn/
│           ├── test_dwn_thermometer.py
│           ├── test_dwn_lut_layer.py
│           ├── test_dwn_group_sum.py
│           └── test_dwn_transform_pass.py
│
└── docs/tutorials/dwn/
    ├── demo.ipynb                      # End-to-end DWN tutorial
    └── benchmark_results/              # Plots and tables
```

---

## Testing Strategy

### Unit Tests (pytest)

| File | Tests |
|---|---|
| `test_dwn_thermometer.py` | Fit/binarize correctness, shape, edge cases |
| `test_dwn_lut_layer.py` | Forward pass shape, gradient flow (non-zero grads), LUT content export |
| `test_dwn_group_sum.py` | Popcount correctness, temperature effect |
| `test_dwn_transform_pass.py` | LFC/MLP model conversion, module replacement, pass_args parsing |

### Integration Tests

- Train DWN on MNIST for 5 epochs, assert accuracy > 90%
- Export LUT contents and verify against reference implementation
- Verify Verilog simulation output matches PyTorch inference

### Hardware Tests (cocotb)

- `fixed_dwn_thermometer_tb.py`: Verify comparator outputs match Python reference
- `fixed_dwn_lut_layer_tb.py`: Random input vectors, compare with Python golden model
- `fixed_dwn_groupsum_tb.py`: Verify popcount outputs

---

## Extension Targets

### Pipeline Registers per Layer

**Motivation:** LUT layers are likely the timing critical path. Pipeline registers at layer boundaries allow higher clock frequency.

**Implementation:**
- Add `PIPELINE_STAGES` parameter to `fixed_dwn_lut_layer.sv`
- Insert `always_ff` register stages between the combinational LUT array and the output
- Adds `L × PIPELINE_STAGES` cycles of latency but enables higher Fmax

**Tradeoff:** 1 pipeline register per layer: typically +50–100 MHz Fmax on Xilinx Artix-7/UltraScale+

### Pipelined Adder Tree in GroupSum

**Motivation:** For very large W/C (e.g., 4000 LUTs / 10 classes = 400 bits to count), combinational `$countones` may not meet timing.

**Implementation:**
Instantiate a parameterised binary adder tree (see the commented-out code in `fixed_difflogic_groupsum.sv` from PR #276 for the template) controlled by `USE_PIPELINED_ADDER_TREE`.

**Tradeoff:** Adds log2(W/C) pipeline stages; unlikely to be bottleneck for W ≤ 4000.

### Conv2D DWN Layers

Current plan focuses on fully-connected (linear) layers only. Extension to conv layers would enable DWN on image tasks without flattening, using spatially-local LUT connectivity.

---

## Implementation Priority

| Phase | Deliverable | Priority |
|---|---|---|
| 1a | `DistributiveThermometer`, `LUTLayer`, `GroupSum` PyTorch modules | **P0** |
| 1b | `DWNModel` class + training script (MNIST) | **P0** |
| 2a | `dwn_transform_pass` for LFC model | **P1** |
| 2b | TOML config + registration in PASSES dict | **P1** |
| 3a | `fixed_dwn_lut_neuron.sv` + `fixed_dwn_lut_layer.sv` | **P1** |
| 3b | `fixed_dwn_groupsum.sv` | **P1** |
| 3c | `fixed_dwn_thermometer.sv` | **P2** |
| 3d | Hardware metadata pass + Verilog emit integration | **P2** |
| 4a | Accuracy benchmark: DWN vs DiffLogicNet on MNIST | **P1** |
| 4b | Parameter count comparison | **P1** |
| 4c | Vivado synthesis: area-delay product comparison | **P2** |
| 4d | Full parameter sweep + training time analysis | **P3** |
| Ext | Pipeline registers, pipelined adder tree | **P3** |

---

## References

1. Bacellar, A. et al. "Differentiable Weightless Neural Networks." *ICML 2024*. arXiv:2410.11112
2. Petersen, F. et al. "Deep Differentiable Logic Gate Networks." *NeurIPS 2022*. arXiv:2210.08277
3. Martha, et al. "NeuraLUT-Assemble: Mixed-Precision LUT-Based Neural Networks." arXiv:2504.00592
4. Umuroglu, Y. et al. "FINN: A Framework for Fast, Scalable Binarized Neural Network Inference." *FPGA 2017*.
5. MASE DiffLogic PR #276: https://github.com/DeepWok/mase/pull/276
6. `torch-dwn` reference implementation: https://github.com/alanbacellar/DWN
