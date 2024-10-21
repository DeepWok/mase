
.. image:: ../../imgs/deepwok.png
   :width: 160px
   :height: 160px
   :scale: 100 %
   :alt: logo
   :align: center

Lab 4 (Hardware Stream) for Advanced Deep Learning Systems (ADLS)
~~~~~

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

In this lab, you will learn how to use the search functionality in the
software stack of MASE to implement a Network Architecture Search.

There are in total 4 tasks you would need to finish, there is also 1
optional task.

Most of the guidance are in
`lab4-hardware.ipynb <https://github.com/DeepWok/mase/blob/main/docs/labs/lab4-hardware.ipynb>`__, make sure you followed
it closely and try to finish the following tasks and answer the
questions.

Tasks
=====

1. Read `this
   page <https://jianyicheng.github.io/mase-tools/modules/analysis/add_metadata.html#add-hardware-metadata-analysis-pass>`__
   for more information on the hardware metadata pass. Why we have such
   a metadata setup? How is this different from the software metadata?

2. Read through ``top/hardware/rtl/top.sv`` and make sure you understand
   how our MLP model maps to this hardware design. Explain what each
   part is doing in the ``.sv`` file.

3. Launch the simulation, log and show the simulation results.

Extension Task
==============

Choose another layer type from the `Pytorch
list <https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity>`__
and write a SystemVerilog file to implement that layer in hardware.
Then, change the generated ``top.sv`` file to inject that layer within
the design. For example, you may replace the ReLU activations with
`Leaky
ReLU <https://pytorch.org/docs/stable/generated/torch.nn.RReLU.html#torch.nn.RReLU>`__.
Re-build the simulation and observe the effect on latency and accuracy.
