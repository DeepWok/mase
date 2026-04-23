Installation
=============================

To use MASE, you can set up the environment using one of the following methods. ``uv`` is recommended for local development, and Docker for a fully isolated environment.

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

    * Exising Pytorch models can be patched to remove control and run symbolic tracing (see `here <https://github.com/DeepWok/mase/tree/main/src/chop/models/patched>`__ for examples).

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

