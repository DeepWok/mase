Tutorials
=============================

Common Usage Examples
---------------------

.. toctree::
    :maxdepth: 1

    :caption: Common Use Cases
    tutorials/common/bert_emit

Interactive Usage with Python
-----------------------------

You can directly import MASE as a package using `import mase` after installation

.. toctree::
    :maxdepth: 1

    :caption: Interactive Usage with Python (Recommended)
    tutorials/common/interactive

The MASE Command Line System
----------------------------

MASE supported a range of `actions` (eg. `train`, `test`, `transform` ...), these are designed to help you to quickly spin things in your command line without touching the source code.

.. toctree::
    :maxdepth: 1

    :caption: The MASE Command Line System
    tutorials/actions/train/simple_train_flow
    tutorials/actions/transform/cli_transform
    tutorials/actions/search/mixed_precision_search_on_mase_graph
    tutorials/actions/search/mixed_precision_search_on_manual_model

The MASE Pass System
--------------------

.. toctree::
    :maxdepth: 1

    :caption: Passes
    tutorials/passes/graph_and_pass
    tutorials/stat_to_config

Advanced Topics
---------------

.. toctree::
    :maxdepth: 1

    :caption: Advanced Topics
    tutorials/Add-model-to-machop
    tutorials/doc_writing
    tutorials/passes/interface/masert/tensorrt/tensorRT_quantization_tutorial
    tutorials/passes/interface/masert/onnxrt/onnxrt_quantization_tutorial
    tutorials/actions/search/how_to_extend_search
    tutorials/accelerate_fsdp
