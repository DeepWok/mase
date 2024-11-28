chop.passes
============================


All passes, no matter analysis or transform, take a standard form:

.. code-block:: python 

  # pass_args is a dict
  def pass(m, pass_args):
      ...
  # info a a dict
  return m, info



Mase Module Analysis Passes
----------------------------

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

  module_analysis/quantization

Mase Module Transform Passes
----------------------------

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

  module_transform/quantization


MaseGraph Analysis Passes
-------------------------

.. list-table:: MaseGraph graph analysis passes 
  :widths: 20 40 40
  :header-rows: 1

  * - Pass Name
    - Usage Example
    - Summary
  * - :py:meth:`~chop.passes.graph.analysis.init_metadata.init_metadata_analysis_pass`
    - `test_add_common_metadata <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/add_metadata/test_add_common_metadata.py>`_
    - Initialize each node with the MaseMetadata, this needs to run first before adding any metadata
  * - :py:meth:`~chop.passes.graph.analysis.add_metadata.add_common_metadata.add_common_metadata_analysis_pass`
    - `test_add_common_metadata <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/add_metadata/test_add_common_metadata.py>`_
    - Add metadata used for both software and hardware, this needs to run first before calling to add_software_metadata or add_hardware_metadata
  * - :py:meth:`~chop.passes.graph.analysis.add_metadata.add_software_metadata.add_software_metadata_analysis_pass`
    - `test_add_software_metadata <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/add_metadata/test_add_software_metadata.py>`_
    - Add hardware-specific metadata, such as which hardware IPs to map and hardware parameters. Details see the pass page
  * - :py:meth:`~chop.passes.graph.analysis.add_metadata.add_hardware_metadata.add_hardware_metadata_analysis_pass`
    - `test_add_hardware_metadata <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/add_metadata/test_add_hardware_metadata.py>`_
    - Add software-specific metadata, such as sparsity. Details see the pass page
  * - :py:meth:`~chop.passes.graph.analysis.report.report_graph.report_graph_analysis_pass`
    - `test_report_graph <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/report/test_report_graph_analysis_pass.py>`_
    - Generates a report for the graph analysis and prints out an over the model in a table.
  * - :py:meth:`~chop.passes.graph.analysis.report.report_node.report_node_hardware_type_analysis_pass`
    - `test_report_node_hardware_type <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/report/test_report_node_hardware_type_analysis_pass.py>`_
    - Generate a report on the hardware type of the nodes in the graph.
  * - :py:meth:`~chop.passes.graph.analysis.report.report_node.report_node_meta_param_analysis_pass`
    - `test_report_node_meta_param <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/report/test_report_node_meta_param_analysis_pass.py>`_
    - Generate a report on the meta parameters of the nodes in the graph.
  * - :py:meth:`~chop.passes.graph.analysis.report.report_node.report_node_shape_analysis_pass`
    - `test_report_node_shape <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/report/test_report_node_shape_analysis_pass.py>`_
    - Generate a report on the shape of the nodes in the graph.
  * - :py:meth:`~chop.passes.graph.analysis.report.report_node.report_node_type_analysis_pass`
    - `test_report_node_type <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/report/test_report_node_type_analysis_pass.py>`_
    - Generate a report on the type of the nodes in the graph.
  * - :py:meth:`~chop.passes.graph.analysis.statistical_profiler.profile_statistics.profile_statistics_analysis_pass`
    - `test_profile_statistics <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/statistic_profiler/test_statistic_profiler.py>`_
    - Perform profile statistics analysis on the given graph
  * - :py:meth:`~chop.passes.graph.analysis.quantization.calculate_avg_bits.calculate_avg_bits_mg_analysis_pass`
    - `calculate_avg_bits_mg_analysis_pass <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/quantization/calculate_avg_bits_mg_analysis_pass.py>`_
    - Calculate, on average, how many bits are spent on weights and activations
  * - :py:meth:`~chop.passes.graph.analysis.pruning.calculate_natural_sparsity.add_natural_sparsity_metadata_analysis_pass`
    - `test_calculate_natural_sparsity <https://github.com/DeepWok/mase/blob/main/test/passes/graph/analysis/pruning/test_calculate_natural_sparsity.py>`_
    - Add natural sparsity metadata analysis pass to the given MaseGraph.
  * - :py:meth:`~chop.passes.graph.analysis.pruning.calculate_sparsity.add_pruning_metadata_analysis_pass`
    - `test_prune <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/prune/test_prune.py>`_
    - This pass computes weight and activation sparsity based on pruning masks
  * - :py:meth:`~chop.passes.graph.analysis.pruning.hook_inspector.hook_inspection_analysis_pass`
    - `test_hook_inspection_analysis_pass <https://github.com/DeepWok/mase/blob/main/test/passes/graph/anlysis/pruning/test_hook_inspect.py>`_
    - Provide hook information of the modules
  * - :py:meth:`~chop.passes.graph.analysis.runtime.runtime_analysis.runtime_analysis_pass`
    - fill me
    - Perform runtime analysis on the given graph (MaseGraph, TensorRT, ONNX models)
  * - :py:meth:`~chop.passes.graph.analysis.verify.verify.verify_metadata_analysis_pass`
    - fill me
    - fill me
  * - :py:meth:`~chop.passes.graph.analysis.verify.verify.verify_common_metadata_analysis_pass`
    - fill me
    - fill me
  * - :py:meth:`~chop.passes.graph.analysis.verify.verify.verify_software_metadata_analysis_pass`
    - fill me
    - fill me
  * - :py:meth:`~chop.passes.graph.analysis.verify.verify.verify_hardware_metadata_analysis_pass`
    - fill me
    - fill me

.. toctree::
  :maxdepth: 2

  analysis/add_metadata
  analysis/autosharding
  analysis/init_metadata
  analysis/report
  analysis/statistical_profiler
  analysis/verify
  analysis/quantization
  analysis/pruning
  analysis/runtime


MaseGraph Transform Passes
--------------------------

.. list-table:: MaseGraph graph transform passes 
  :widths: 20 40 40
  :header-rows: 1

  * - Pass Name
    - Usage Example
    - Summary
  * - :py:meth:`~chop.passes.graph.transforms.pruning.prune.prune_transform_pass`
    - `test_prune <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/prune/test_prune.py>`_
    - Prune the given graph
  * - :py:meth:`~chop.passes.graph.transforms.pruning.prune_detach_hook.prune_detach_hook_transform_pass`
    - `test_prune_detach_hook <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/prune/test_prune_detach_hook.py>`_
    - Remove all pruning hooks
  * - :py:meth:`~chop.passes.graph.transforms.quantize.quantize_transform_pass`
    - `test_quantize <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/quantize/test_quantize.py>`_
    - Apply quantization, check the further documentation below
  * - :py:meth:`~chop.passes.graph.transforms.quantize.summarize_quantization_analysis_pass`
    - `test_quantize <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/quantize/test_quantize.py>`_
    - Summarizes the quantization with respect to the original graph
  * - :py:meth:`~chop.passes.graph.transforms.utils.conv_bn_fusion_transform_pass`
    - `test_conv_bn_fusion <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/utils/test_conv_bn_fusion.py>`_
    - Perform Conv-BN fusion on the given graph 
  * - :py:meth:`~chop.passes.graph.transforms.utils.logicnets_fusion_transform_pass`
    - `test_logicnets_fusion <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/utils/test_conv_bn_fusion.py>`_ (DEV, Disabled) 
    - Perform LogicNets fusion on the given graph (DEV, Disabled) 
  * - :py:meth:`~chop.passes.graph.transforms.utils.onnx_annotate_transform_pass`
    - NA (DEV, Disabled) 
    - Convert MaseGraph to ONNX and annotate the relevant layers with sparsity information (DEV, Disabled) 
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_top.emit_verilog_top_transform_pass`
    - `test_emit_verilog_linear <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/verilog/test_test_emit_verilog_linear.py>`_
    - Emit the top-level model design in Verilog.
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_tb.emit_verilog_tb_transform_pass`
    - Emit test bench and related files for simulation.
    - `test_emit_verilog_linear <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/verilog/test_test_emit_verilog_linear.py>`_
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_bram.emit_bram_transform_pass`
    - Enumerate input parameters of the node and emit a ROM block with handshake interface for each parameter
    - `test_emit_verilog_linear <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/verilog/test_test_emit_verilog_linear.py>`_
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_top.emit_mlir_hls_transform_pass`
    - Emit the hardware components that generated from MLIR-HLS tools. 
    - `test_emit_verilog_linear <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/verilog/test_test_emit_verilog_linear.py>`_
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_top.emit_internal_rtl_transform_pass`
    - Emit the hardware components that pre-defined in the mase internal library.
    - `test_emit_verilog_linear <https://github.com/DeepWok/mase/blob/main/test/passes/graph/transforms/verilog/test_test_emit_verilog_linear.py>`_
  * - :py:meth:`~chop.passes.graph.transforms.tensorrt.quantize.calibrate.tensorrt_fake_quantize_transform_pass`
    - Apply TensorRT fake quantization to the given graph for INT8 quantization calibration
    - fille me
  * - :py:meth:`~chop.passes.graph.transforms.tensorrt.quantize.calibrate.tensorrt_calibrate_transform_pass`
    - Apply TensorRT calibration to the given graph for INT8 quantization
    - fille me
  * - :py:meth:`~chop.passes.graph.transforms.tensorrt.quantize.fine_tune.tensorrt_fine_tune_transform_pass`
    - Apply TensorRT fine tune to the given graph for quantization aware training
    - fille me


.. toctree::
  :maxdepth: 2

  transform/pruning
  transform/quantize
  transform/verilog
  transform/utils
  transform/tensorrt



MaseGraph Interface Passes
--------------------------

.. list-table:: A summary of all MaseGraph interface passes 
  :widths: 20 40 40
  :header-rows: 1

  * - Pass Name
    - Usage Example
    - Summary
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.load_mase_graph_interface_pass`
    - fill me
    - fill me
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.save_mase_graph_interface_pass`
    - fill me
    - fill me
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.save_node_meta_param_interface_pass`
    - fill me
    - fill me
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.load_node_meta_param_interface_pass`
    - fill me
    - fill me
  * - :py:meth:`~chop.passes.graph.interface.tensorrt.quantize.tensorrt_engine_interface_pass`
    - Converts the given graph to a TensorRT engine model
    - fill me
  * - :py:meth:`~chop.passes.graph.interface.onnxrt.onnx_runtime.onnx_runtime_interface_pass`
    - Converts the given graph to a ONNXRuntime model
    - fill me

.. toctree::
  :maxdepth: 2

  interface/save_and_load
  interface/tensorrt
  interface/onnxrt


Additional Documentation
------------------------

.. toctree::
	:maxdepth: 1

	transform/quantize_further_doc

.. test-results:: ../../software_report.xml
