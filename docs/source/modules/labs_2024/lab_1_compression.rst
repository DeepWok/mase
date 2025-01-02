
.. image:: ../../imgs/deepwok.png
   :width: 160px
   :height: 160px
   :scale: 100 %
   :alt: logo
   :align: center

Lab 1: Model Compression (Quantization and Pruning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div align="center">
   <p align="center">
      ELEC70109/EE9-AML3-10/EE9-AO25
      <br />
   Written by
      <a href="https://aaron-zhao123.github.io/">Aaron Zhao </a> and
      <a href="https://www.pedrogimenes.co.uk/">Pedro Gimenes </a>
   </p>
   </div>


General introduction
=====================

In this lab, you will learn how to use Mase to compress a Bert model using quantization and pruning. You will build off the checkpoint from Lab 2, where we fine tuned a Bert model for sequence classification using the LoRA adapter. You will quantize the model to fixed-point precision and then prune the model to reduce the number of parameters. After each stage, you'll run further fine tuning to recover the performance lost during compression.

Learning tasks
=============================

1. Go through `"Tutorial 3: Running Quantization-Aware Training (QAT) on Bert" <https://github.com/DeepWok/mase/blob/main/docs/source/modules/documentation/tutorials/tutorial_3_qat.ipynb>`__ to learn how to quantize the Bert model and run post-quantization finetuning.

2. Go through `"Tutorial 4: Unstructured Pruning on Bert" <https://github.com/DeepWok/mase/blob/main/docs/source/modules/documentation/tutorials/tutorial_4_pruning.ipynb>`__ to understand how to prune a quantized model for further compression.

Implementation tasks
=============================

1. In Tutorial 3, you quantized every Linear layer in the model to the provided configuration. Now, explore a range of fixed point widths from 4 to 32. 

   a. Plot a figure where the x-axis is the fixed point width and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 3. 
   
   b. Plot separate curves for PTQ and QAT at each precision to show the effect of post-quantization finetuning.

2. Take your best obtained model from Task 1 and rerun the pruning procedure, this time varying the sparsity from 0.1 to 0.9. 

   a. Plot a figure where the x-axis is the sparsity and the y-axis is the highest achieved accuracy on the IMDb dataset, following the procedure in Tutorial 4. 
   
   b. Plot separate curves for ``Random`` and ``L1-Norm`` methods to evaluate the effect of different pruning strategies.