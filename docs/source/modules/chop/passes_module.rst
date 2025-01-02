chop.passes.module
============================


Summary of Mase Module Analysis Passes
--------------------------------------

.. list-table:: MASE module-level analysis passes 
  :widths: 20 40 40
  :header-rows: 1

  * - Pass Name
    - Usage Example
    - Summary
  * - :py:meth:`~chop.passes.module.analysis.quantize.calculate_avg_bits_module_analysis_pass`
    - `test_calcualte_avg_bits_module <https://github.com/DeepWok/mase/blob/main/test/passes/module/analysis/quantize/test_calcualte_avg_bits_module.py>`_
    - Analyzes the averaged bitwidth of a given module.

.. toctree::
  :maxdepth: 2
  :caption: Full list of module-level analysis passes

  module_analysis/quantization

Summary of Mase Module Transform Passes
---------------------------------------

.. list-table:: MASE module-level transform passes 
  :widths: 20 40 40
  :header-rows: 1

  * - Pass Name
    - Usage Example
    - Summary
  * - :py:meth:`~chop.passes.module.transforms.quantize.quantize_module_transform_pass`
    - `test_module_quantize <https://github.com/DeepWok/mase/blob/main/test/passes/module/transforms/quantize/test_quantize_module.py>`_
    - Apply quantization transformation to the given nn.Module

.. toctree::
  :maxdepth: 2
  :caption: Full list of module-level transform passes

  module_transform/quantization