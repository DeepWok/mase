
.. image:: ../../imgs/deepwok.png
   :width: 160px
   :height: 160px
   :scale: 100 %
   :alt: logo
   :align: center

Lab 1: Introduction to Mase
~~~~~

.. raw:: html

   <div align="center">
   <p align="center">
      ELEC70109/EE9-AML3-10/EE9-AO25
      <br />
   Written by
      <a href="https://aaron-zhao123.github.io/">Aaron Zhao </a> ,
      <a href="https://www.pedrogimenes.co.uk/">Pedro Gimenes </a> ,
      <a href="https://chengzhang-98.github.io/blog/">Cheng Zhang </a>
   </p>
   </div>

General introduction
====================

In this lab, you will learn how to use the basic functionalities of Mase. You will be required to run through some of the `tutorials <https://deepwok.github.io/mase/modules/documentation/tutorials.html>`__ in the documentation, which introduce you to the fundamental aspects of the framework, including:

1. Importing models into the framework and generating a compute graph
2. Understanding the Mase IR and its benefit over other ways of representing Machine Learning workloads
3. Writing and executing Torch FX passes to optimize a model

You will start by generating a MaseGraph for a Bert model. You will then fine tune this model using a LoRA adapter to achieve high performance on the IMDB sequence classification dataset. In future labs, you will build off this work to explore more advanced features of the MASE framework.

Learning tasks
=============================

1. Make sure you have read and understood the installation of the framework, detailed `here <https://deepwok.github.io/mase/modules/documentation/getting_started.html>`__.

2. Go through `"Tutorial 1: Introduction to the Mase IR, MaseGraph and Torch FX passes" <https://github.com/DeepWok/mase/blob/adls_2024/docs/source/modules/documentation/tutorials/tutorial_1_introduction_to_mase.ipynb>`__ to understand the basic concepts of the framework.

3. Go through `"Tutorial 2: Insert a LoRA adapter to Finetune Bert for Sequence Classification" <https://github.com/DeepWok/mase/blob/adls_2024/docs/source/modules/documentation/tutorials/tutorial_2_lora_finetune.ipynb>`__ to understand how to fine-tune a model using the LoRA adapter.

Implementation tasks
=============================

1. In Tutorial 1, you saw how to a transform pass to remove dropout nodes. Now, write another transform pass to insert...

2. In Tutorial 2, you saw how to fine-tune a Bert model using the LoRA adapter. Now, extend the `insert_lora_adapter_transform_pass` to support one of the following models. Repeat the procedure in the tutorial to evaluate the model on the IMDB dataset.

      - RoBERTa

      - ...
