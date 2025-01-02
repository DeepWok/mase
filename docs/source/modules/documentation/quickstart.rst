Quickstart
=============================

This page is a draft and will be finished when the pipeline feature is completed.
For now, you can following the tutorials to get started with Mase.

Importing a model
--------------------------------

.. code-block:: python

    from transformers import AutoModel

    model = AutoModel.from_pretrained("bert-base-uncased")
    mg = MaseGraph(model)

Architecture Search
--------------------------------

When you're unsure about the best architecture for your model, you can use the ``search`` action, which deploys a number of search strategies to find the optimal architecture that maximises performance on a given dataset.

.. code-block:: python

    from chop.actions.search

    dataset = ...
    metric = ...

    model = search(dataset, metric)

Pipelines
--------------------------------

Mase contains a collection of compiler passes with a range of functions. For common use cases, ``Pipelines`` encapsulate all passes required to quickly achieve a specific goal.

Automatic Kernel Fusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For optimized inference on GPUs, you can use the ``PipelineForKernelFusion``. This pipeline finds operator fusion opportunities in the model and applies them to reduce the number of GPU kernel launches.

.. code-block:: python

    from chop.pipelines import PipelineForKernelFusion

    pipe = PipelineForKernelFusion()

    mg, _ = pipe(mg)

Model Compression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``PipelineForCompression`` prunes the model and applies mixed-precision search to find the optimal quantization configuration.

.. code-block:: python

    from chop.pipelines import PipelineForCompression

    pipe = PipelineForCompression()

    mg, _ = pipe(mg)

Generate an FPGA accelerator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can run the ``PipelineForEmitVerilog`` after pruning and quantizing the model to generate an FPGA accelerator.

.. code-block:: python

    from chop.pipelines import PipelineForEmitVerilog

    pipe = PipelineForEmitVerilog()

    mg, _ = pipe(mg)
