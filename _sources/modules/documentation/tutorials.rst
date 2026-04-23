Tutorials
=============================

The following tutorials walk through the main flows of MASE, from model training to deployment.

.. hint::

   For a quick introduction, see the Quickstart page. The tutorials on this page dive deeper into various features and use cases.

See below a summary of the main tutorials:

.. figure:: tutorials/imgs/tutorial_overview.png
   :width: 80%
   :align: center

   Overview of the main Mase tutorials.

Core Tutorials
---------------------

The following tutorials show how to use Mase for standard model compression and optimization flows, such as Finetuning, Quantization and Pruning.

.. toctree::
   :maxdepth: 1

   tutorials/tutorial_1_introduction_to_mase_script
   tutorials/tutorial_2_lora_finetune_script
   tutorials/tutorial_3_qat_script
   tutorials/tutorial_4_pruning_script

Architecture Search
---------------------

The following show how to effectively run Neural Architecture Search (NAS) and mixed-precision search to find optimal architecture configurations for a given task or dataset.

.. toctree::
   :maxdepth: 1

   tutorials/tutorial_5_nas_optuna_script
   tutorials/tutorial_6_mixed_precision_search_script


Developer Guide
---------------

If you'd like to contribute to Mase, you may find the following resources useful.

.. toctree::
   :titlesonly:

   tutorials/developer/Add-model-to-chop
   tutorials/developer/doc_writing
   tutorials/developer/how_to_extend_search

