Tutorial 1: Introduction to the Mase IR, MaseGraph and Torch FX passes
=======================================================================

This tutorial is maintained as a plain Python script.

- Student run command: ``uv run python docs/source/modules/documentation/tutorials/tutorial_1_introduction_to_mase.py``
- Single source of truth: ``docs/source/modules/documentation/tutorials/tutorial_1_introduction_to_mase.py``
- This page only documents the flow and references code with ``literalinclude``.

Run this tutorial
-----------------

From the repository root:

.. code-block:: bash

   uv run python docs/source/modules/documentation/tutorials/tutorial_1_introduction_to_mase.py

Step 1: Load a pretrained model
-------------------------------

.. literalinclude:: tutorial_1_introduction_to_mase.py
   :language: python
   :start-after: # [load_model:start]
   :end-before: # [load_model:end]

Step 2: Build the FX graph
--------------------------

.. literalinclude:: tutorial_1_introduction_to_mase.py
   :language: python
   :start-after: # [build_graph:start]
   :end-before: # [build_graph:end]

Step 3: Raise FX graph to Mase IR
---------------------------------

.. literalinclude:: tutorial_1_introduction_to_mase.py
   :language: python
   :start-after: # [raise_ir:start]
   :end-before: # [raise_ir:end]

Step 4: Write and run an analysis pass
--------------------------------------

.. literalinclude:: tutorial_1_introduction_to_mase.py
   :language: python
   :pyobject: count_dropout_analysis_pass

.. literalinclude:: tutorial_1_introduction_to_mase.py
   :language: python
   :start-after: # [analysis_pass:start]
   :end-before: # [analysis_pass:end]

Step 5: Write and run a transform pass
--------------------------------------

.. literalinclude:: tutorial_1_introduction_to_mase.py
   :language: python
   :pyobject: remove_dropout_transform_pass

.. literalinclude:: tutorial_1_introduction_to_mase.py
   :language: python
   :start-after: # [transform_pass:start]
   :end-before: # [transform_pass:end]

Step 6: Export and reload MaseGraph
-----------------------------------

.. literalinclude:: tutorial_1_introduction_to_mase.py
   :language: python
   :start-after: # [export:start]
   :end-before: # [export:end]

