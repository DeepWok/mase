
.. image:: ../../imgs/deepwok.png
   :width: 160px
   :height: 160px
   :scale: 100 %
   :alt: logo
   :align: center

Lab 3: Mixed Precision Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
====================

You have looked at how to quantize models in lab0 and how to search for optimal architectures in lab2. In this lab, you will learn how to use Mase to search for optimal quantization schemes for a model.

Learning tasks
=============================

1. Go through `"Tutorial 6: Mixed Precision Quantization Search with Mase and Optuna" <https://github.com/DeepWok/mase/blob/main/docs/source/modules/documentation/tutorials/tutorial_6_mixed_precision_search.ipynb>`__ to understand how to use Mase to search for optimal quantization schemes for a model.

Implementation tasks
=============================

1. In Tutorial 6, all layers allocated to IntegerLinear are allocated the same width and fractional width. This is suboptimal, as different layers may have different sensitivities to quantization. 
   
   a. Modify the code to allow different layers to have widths in the range [8, 16, 32] and fractional widths in the range [2, 4, 8]. Expose this choice as an additional hyperparameter for the Optuna sampler. 
   
   b. Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis.
   
2. In Section 1 of Tutorial 6, when defining the search space, a number of layers are imported, however only LinearInteger and the full precision nn.Linear are selected. 

   a. Now, extend the search to consider all supported precisions for the Linear layer in Mase, including Minifloat, BlockFP, BlockLog, Binary, etc. This may also require changing the model constructor so the required arguments are passed when instantiating each layer.

   b. Run the search again, and plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. Plot one curve for each precision to compare their performance.