Installation
=============================

To use MASE, you can easily set up an environment with all required dependencies using either uv or Docker. Follow the instructions at the links below according to your preferred method.
The model used in the following tutorials runs within a reasonable time on a CPU machine (under 10 minutes on a MacBook Air M4). Faster performance can be achieved on a local GPU machine, a university server, or Google Colab.
.. hint::

    Some parts of the flow may assume you have a version of Vivado/Vitis installed. Other parts, such as the emit verilog flow, require Verilator, which is included in the Docker container, but not on the uv environment. If you prefer using uv, you can just install Verilator locally.

.. toctree::
    :maxdepth: 1

    getting_started/Get-started-using-uv
    getting_started/Get-started-using-Docker
    getting_started/Get-started-students

Import a model
----------------

To import a model into MASE and use all its features, the following options are available.

* Generate a MaseGraph from a torch.nn.Module instance.

    * Since the MASE IR is based on `Torch FX <https://pytorch.org/docs/stable/fx.html>`_, symbolic tracing limitations apply to this method, namely models with control flow cannot be traced (see `documentation <https://pytorch.org/docs/stable/fx.html#limitations-of-symbolic-tracing>`_).

    * Exising Pytorch models can be patched to remove control and run symbolic tracing (see `here <https://github.com/DeepWok/mase/tree/main/machop/chop/models/patched>`__ for examples).

.. code-block:: python

    import torch.nn as nn
    from chop import MaseGraph

    class MyModel(nn.Module):
        def __init__(self):
            ...

        def forward(self):
            ...

    model = MyModel()
    mg = MaseGraph(model)

* Import a model using the ONNX backend.

    * HuggingFace models can be imported using the :code:`MaseOnnxGraph.from_pretrained` method, which first exports an ONNX representation of the model, then imports it into MASE. See list of supported models `here <https://huggingface.co/docs/optimum/en/exporters/onnx/overview>`_.

    * Any other ONNX models can be directly imported using the :code:`export_fx_graph` analysis pass.

.. code-block:: python

    from chop import MaseOnnxGraph
    from chop.passes export_fx_graph_analysis_pass

    pretrained = "bert-base-uncased"
    og = MaseOnnxGraph.from_pretrained(pretrained)
    mg, _ = export_fx_graph_analysis_pass(og)
