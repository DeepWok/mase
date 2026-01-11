
.. image:: ../../imgs/deepwok.png
   :width: 160px
   :height: 160px
   :scale: 100 %
   :alt: logo
   :align: center

Lab 4 (Software Stream) Performance Engineering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div align="center">
   <p align="center">
      ELEC70109/EE9-AML3-10/EE9-AO25
      <br />
   Written by
      <a href="https://aaron-zhao123.github.io/">Aaron Zhao </a> ,
      <a href="https://chengzhang-98.github.io/blog/">Cheng Zhang </a> ,
      <a href="https://www.pedrogimenes.co.uk/">Pedro Gimenes </a>
   </p>
   </div>

General introduction
====================

In this lab, you will learn how to optimize ML code. We will go through three approaches:

1. Use high-level framework such as `torch.compile` to optimize user code, and understand the basic building blocks in such optimization frameworks.
2. Understand the effect of fused kernels, and test it with existing upstream implementations in Pytorch.
3. Understand how to port custom CUDA kernels into Pytorch, and test their performances.


Learning tasks
=============================

1. Go through `"Lab 4 for Advanced Deep Learning Systems (ADLS) - Software Stream" <https://github.com/DeepWok/mase/blob/main/docs/labs/lab4-software.ipynb>`__ to understand how to use optimize your ML model.

Implementation tasks
=============================

1. In the first part of Lab 4 (`torch.compile`), we did not really observe real run-time speedups with `torch.compile`.

   a. Modify the code and investigate why this is the case?

   b. If you change the `device` to `cuda`, do you observe the same thing?

2. In the second part of Lab 4 (kernel fusion), we looked at a fused SDPA kernel.

   a. Now, extend the profiling to the SDPA kernel, compare its runtime behavior with the naive implementation.

   b. If you change the `device` to `cuda`, do you observe the same thing?

3. In the third part of lab4 (Custom kernel), we go through how to write MXINT8 dequantization kernel and bind it to Python.

   a. How does MXINT8 benefit custom hardware if both the activation and weights in a linear layer are quantized to MXINT8?

   b. What is the purpose of the variable `dont_need_abs` and `bias` in the C++ for loop?

   c. How does `cta_tiler` partition data for copying to shared memory in CUDA kernel? How does `layout_sX` partition threads in a threadlock for computation? (Challenge)

   d. Why the saved GPU memory is not exactly (32 - (4+8/32))/32 = 86.7% of the FP32 model?