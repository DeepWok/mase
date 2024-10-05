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
    tutorials/common/interactive_quantize

The MASE Pass System
--------------------


One specific feature of MASE is its capability to transform DL models to a computation graph and apply analysis or transformation passes on the graph.

In MASE, we support two granularities on these passes, namely `module` level and `graph` level:

- `module` level passes: this only provides coarse-grained control over any `torch.nn.Module` in the model. We refer this as Module Passes.
- `graph` level passes: this provides fine-grained control over the computation graph of the model using the `torch.fx <https://pytorch.org/docs/stable/fx.html>`_ framework. This also nessitates the use of `torch.fx` to represent the computation graph.  We provide a set of APIs to convert a model to a `MASEGraph`, starting from the `Notebook <https://github.com/DeepWok/mase/blob/main/docs/labs/lab2_transform_no_CLI.ipynb>`_.

These tutorials will guide you through the process of creating and applying passes.

.. toctree::
    :maxdepth: 1

    tutorials/passes/module_and_pass
    tutorials/passes/graph_and_pass

You may have realized that MASE, especially when used with the pass system on the graph-level, has a few customized terminologies. We call this the MASE IR system. The following tutorial will guide you through the MASE IR system.

.. toctree::
    :maxdepth: 1

    tutorials/passes/mase_ir


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
