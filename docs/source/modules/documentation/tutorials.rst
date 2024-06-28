Tutorials
=============================

Common Usage Examples
---------------------

.. toctree::
    :maxdepth: 1

    tutorials/common/bert_emit

Interactive Usage with Python
-----------------------------

You can directly import MASE as a package using `import mase` after installation

.. toctree::
    :maxdepth: 1

    tutorials/common/interactive

The MASE Command Line System
----------------------------

MASE supported a range of `actions` (eg. `train`, `test`, `transform` ...), these are designed to help you to quickly spin things in your command line without touching the source code.

.. toctree::
    :maxdepth: 1

    tutorials/actions/train/simple_train_flow
    tutorials/actions/transform/cli_transform
    tutorials/actions/search/mixed_precision_search_on_mase_graph
    tutorials/actions/search/mixed_precision_search_on_manual_model

The MASE Pass System
--------------------

.. toctree::
    :maxdepth: 1

    tutorials/passes/graph_and_pass

Advanced Topics
---------------

.. toctree::
    :maxdepth: 1

    tutorials/Add-model-to-machop
    tutorials/stat_to_ternary_q_config
    tutorials/doc_writing
    tutorials/passes/interface/masert/tensorrt/tensorRT_quantization_tutorial
    tutorials/passes/interface/masert/onnxrt/onnxrt_quantization_tutorial
    tutorials/actions/search/how_to_extend_search
    tutorials/accelerate_fsdp
