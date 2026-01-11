chop.passes
============================


All passes, no matter analysis or transform, take a standard form:

.. code-block:: python 

  # pass_args is a dict
  def pass(m, pass_args):
      ...
  # info a a dict
  return m, info


.. toctree::
    :maxdepth: 1
    :caption: Module-level passes

    passes_module

.. toctree::
    :maxdepth: 1
    :caption: Graph-level passes

    passes_graph