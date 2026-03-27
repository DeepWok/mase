# Necessary things for a good report

- figures that explain the system design
- code snippets
- utilisation report

# 1) Introduction

- mase emit hardware flow is great <glaze the guy>
- the current capabilities for FPGA emit is limited to storing all weights and biases on chip
- for the specific cases where we want to put a large model onto cheap hardware, we are limited by BRAM
- to add functionality to what mase hardware emit can do, we propose our following implementation towards off chip compatible hardware generation, alongside testbenches
- we test this with a specific use case inspired by collaboration with UCL's Optical Networks Group


# 2) Technical overview

## 2a) How mase currently works

- describe top file generation. separate classes for emitting interface, modules and interconnects
- describe how dependency files are copied in, this is quite simple
- show how the only option supported is BRAM

## 2b) What we have added

- we have first added the dram transform pass function. it's sole purpose to iterate over all nodes in masegraph, and if there's support for off chip storage of weights and biases, replace it's interface metadata field with DRAM

- from here we have added a lot of conditional logic to the top file emitters and the dependcy files to add support for off chip weights and biases, whilst setting up a simple framework for adding support for more nodes.


### What exactly we did. Technical details

- added to the interface allowing for streaming in of weights and biases alongside the model inputs
- removed the weight/bias source BRAM modules and their interconnects
- added XYZ (what Harun will be generating) to handle the external DRAM/SDRAM control

- these have been integrated into the mase hardware emit flow with minimal changes to repository itself


## 2c Testing 

- we have added extra logic to the simulate function such that it recognizes if the hardware relies on off chip memory and changes its DUT interface and has a separate case.

has been done


# 3) Case study

- we tested this result by training a symbol classifier for 64-QAM
- here are the utilization reports
- using offchip memory reduces the size of the model by X% allowing us to deploy larger models
- the throughput is limited to Y% which is the tradeoff we have to live iwth. however, this can be optimized with some prefetching/caching [cite this]


# Citations

Papers supporting the claim that offchip memory increases the max size of models we can deploy

"A Multi-Cache System for On-Chip Memory Optimization in FPGA-Based CNN Accelerators" (MDPI, 2021): Details how moving weights off-chip drastically cuts memory usage. It highlights the resulting performance bottleneck.

"Compute-In-Memory on FPGAs for Deep Learning: A Review" (ASU, 2025): Explores the massive throughput gap between on-chip BRAM compute and off-chip memory fetches.

"Accuracy, Training Time and Hardware Efficiency Trade-Offs for Quantized Neural Networks on FPGAs" (ARC, 2020): Quantifies the high latency and energy costs of accessing off-chip DDR compared to BRAM.


# Future work (but don't call it that)

Things to include in a future work section, but since he ain't marking us on that, say that we have "this project is intentioned to be the start of wider build out of support, and we have kept in mind these future expansions"

Smart AXI Burst Controllers

Standard DDR ports often waste cycles. Mase could generate custom memory controllers. These would analyze the tensor shapes from the model. They would then issue perfectly aligned, maximum-length AXI bursts. This saturates the memory bus completely.
Hardware Layer Fusion

Trips to off-chip memory kill performance. Mase could map multiple layers into a single hardware pipeline. You stream the weights for a fused block of layers at once. Intermediate activations stay purely in fast BRAM.

