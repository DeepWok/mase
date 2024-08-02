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

The MASE Command Line System
----------------------------

MASE supported a range of `actions` (eg. `train`, `test`, `transform` ...), these are designed to help you to quickly spin things in your command line without touching the source code.

.. toctree::
    :maxdepth: 1

    tutorials/actions/train/simple_train_flow
    tutorials/actions/transform/cli_transform
    tutorials/actions/search/mixed_precision_search_on_mase_graph
    tutorials/actions/search/mixed_precision_search_on_manual_model

For simplicity, we summarized the following commands for you to quickly start with MASE Command Line System:

.. code-block:: bash

    # Basic training and testing
    ## training
    ./ch train jsc-tiny jsc --max-epochs 3 --batch-size 256 --accelerator cpu --project tmp --debug
    ## test
    ./ch test jsc-tiny jsc --accelerator cpu --debug --load ../mase_output/tmp/software/training_ckpts/best.ckpt --load-type pl

    # Graph-level
    ## transfrom on graph level
    ./ch transform --config $MASE/configs/examples/jsc_toy_by_type.toml --task cls --accelerator=cpu --load ../mase_output/tmp/software/training_ckpts/best.ckpt --load-type pl
    ## search command
    ./ch search --config $MASE/configs/examples/jsc_toy_by_type.toml --task cls --accelerator=cpu --load ../mase_output/tmp/software/training_ckpts/best.ckpt --load-type pl
    ## train searched network
    ./ch train jsc-tiny jsc --max-epochs 3 --batch-size 256 --accelerator cpu --project tmp --debug --load ../mase_output/jsc-tiny/software/transform/transformed_ckpt/graph_module.mz --load-type mz

    # Module-level
    ## transfrom on module level
    ./ch transform --config $MASE/configs/examples/jsc_toy_by_type_module.toml --task cls --accelerator=cpu --load ../mase_output/tmp/software/training_ckpts/best.ckpt --load-type pl
    ## train the transformed network
    ./ch train jsc-tiny jsc --max-epochs 3 --batch-size 256 --accelerator cpu --project tmp --debug --load ../mase_output/jsc-tiny/software/transform/transformed_ckpt/state_dict.pt --load-type pt

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
