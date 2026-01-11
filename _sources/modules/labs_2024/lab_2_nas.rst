
.. image:: ../../imgs/deepwok.png
   :width: 160px
   :height: 160px
   :scale: 100 %
   :alt: logo
   :align: center

Lab 2: Neural Architecture Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

In this lab, you will learn how to use Mase to conduct a Neural Architecture Search (NAS) on a BERT model.

Learning tasks
=============================

1. Go through `"Tutorial 5: Neural Architecture Search (NAS) with Mase and Optuna" <https://github.com/DeepWok/mase/blob/main/docs/source/modules/documentation/tutorials/tutorial_5_nas_optuna.ipynb>`__ to understand how to use high-level hyperparameter optimization frameworks like optuna to conduct a NAS on BERT.

Implementation tasks
=============================

1. Tutorial 5 shows how to use random search to find the optimal configuration of hyperparameters and layer choices for the Bert model. 

   a. Now, explore using the GridSampler and TPESampler in Optuna. 
   
   b. Plot a figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. Plot one curve for each sampler to compare their performance.

2. In Tutorial 5, NAS is used to find an optimal configuration of hyperparameters, then we use the CompressionPipeline in Mase to quantize and prune the model after search is finished. However, the final compressed model may not be optimal, since different model architectures may have different sensitivities to quantization and pruning. Ideally, we want to run a compression-aware search flow, where the quantization and pruning is considered in each trial. 

   a. In the objective function, after the model is constructed and trained for some iterations, call the CompressionPipeline to quantize and prune the model, then continue training for a few more epochs. Use the sampler that yielded the best results in Task 1 to run the compression-aware search. The objective function should return the final accuracy of the model after compression. Consider also the case where final training is performed after quantization/pruning.

   b. Plot a new figure that has the number of trials on the x axis, and the maximum achieved accuracy up to that point on the y axis. There should be three curves: 1. the best performance from Task 1 (without compression), compression-aware search without post-compression training, and compression-aware search with post-compression training.