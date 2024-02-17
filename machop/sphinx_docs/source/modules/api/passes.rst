Chop Passes
============================

All passes, no matter analysis or transform, take a standard form:

.. code-block:: python 

  # pass_args is a dict
  def pass(mg, pass_args):
      ...
  # info a a dict
  return mg, info


MaseGraph Analysis Passes
-------------------------

.. list-table:: A summary of all MaseGraph analysis passes 
  :widths: 25 75
  :header-rows: 1

  * - Pass Name
    - Summary
  * - :py:meth:`~chop.passes.graph.analysis.init_metadata.init_metadata_analysis_pass`
    - Initialize each node with the MaseMetadata, this nees to run first before adding any metadata
  * - :py:meth:`~chop.passes.graph.analysis.add_metadata.add_common_metadata.add_common_metadata_analysis_pass`
    - Add metadata used for both software and hardware, this nees to run first before calling to add_software_metadata or add_hardware_metadata
  * - :py:meth:`~chop.passes.graph.analysis.add_metadata.add_software_metadata.add_software_metadata_analysis_pass`
    - Add hardware-specific metadata
  * - :py:meth:`~chop.passes.graph.analysis.add_metadata.add_hardware_metadata.add_hardware_metadata_analysis_pass`
    - Add software-specific metadata
  * - :py:meth:`~chop.passes.graph.analysis.report.report_graph.report_graph_analysis_pass`
    -  Generates a report for the graph analysis and prints out an over the model in a table.
  * - :py:meth:`~chop.passes.graph.analysis.report.report_node.report_node_hardware_type_analysis_pass`
    -  Perform hardware type analysis on the given graph.
  * - :py:meth:`~chop.passes.graph.analysis.report.report_node.report_node_meta_param_analysis_pass`
    -  Perform meta parameter analysis on the nodes in the graph and generate a report.
  * - :py:meth:`~chop.passes.graph.analysis.report.report_node.report_node_shape_analysis_pass`
    -  Perform shape analysis on the nodes in the graph.
  * - :py:meth:`~chop.passes.graph.analysis.report.report_node.report_node_type_analysis_pass`
    -  Perform a node type analysis on the given graph, pretty print MaseGraph after initialization/loading.
  * - :py:meth:`~chop.passes.graph.analysis.statistical_profiler.profile_statistics.profile_statistics_analysis_pass`
    -  Perform profile statistics analysis on the given graph
  * - :py:meth:`~chop.passes.graph.analysis.quantization.calculate_avg_bits.calculate_avg_bits_mg_analysis_pass`
    -  Calculate, on average, how many bits are spent on weights and activations
  * - :py:meth:`~chop.passes.graph.analysis.pruning.calculate_sparsity.add_pruning_metadata_analysis_pass`
    - Add post-pruning metadata analysis pass to the given graph, the graph must have been pruned
  * - :py:meth:`~chop.passes.graph.analysis.pruning.calculate_natural_sparsity.add_natural_sparsity_metadata_analysis_pass`
    - Add natural sparsity metadata analysis pass to the given MaseGraph.
  * - :py:meth:`~chop.passes.graph.analysis.pruning.hook_inspector.hook_inspection_analysis_pass`
    - Remove and provide hook information of the modules


.. toctree::
	  :maxdepth: 2

	  analysis/add_metadata
	  analysis/init_metadata
	  analysis/report
	  analysis/statistical_profiler
	  analysis/verify
	  analysis/quantization
	  analysis/pruning


MaseGraph Transform Passes
--------------------------

.. list-table:: A summary of all MaseGraph transform passes 
  :widths: 25 75
  :header-rows: 1

  * - Pass Name
    - Summary
  * - :py:meth:`~chop.passes.graph.transforms.pruning.prune.prune_transform_pass`
    - Apply pruning transformation to the given graph
  * - :py:meth:`~chop.passes.graph.transforms.pruning.prune_detach_hook.prune_detach_hook_transform_pass`
    - Apply a transformation to the given graph to remove all pruning hooks
  * - :py:meth:`~chop.passes.graph.transforms.quantize.quantize_transform_pass`
    - Apply quantization transformation to the given graph
  * - :py:meth:`~chop.passes.graph.transforms.quantize.summarize_quantization_analysis_pass`
    - fille me
  * - :py:meth:`~chop.passes.graph.transforms.utils.conv_bn_fusion_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.utils.logicnets_fusion_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.utils.onnx_annotate_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_top.emit_verilog_top_transform_pass`
    - Emit the top-level model design in Verilog
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_tb.emit_verilog_tb_transform_pass`
    - Emit test bench and related files for simulation
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_bram.emit_bram_transform_pass`
    - Enumerate input parameters of the node and emit a ROM block with handshake interface for each parameter
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_top.emit_mlir_hls_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_top.emit_internal_rtl_transform_pass`
    - fill me

.. toctree::
	  :maxdepth: 2

	  transform/pruning
	  transform/quantize
	  transform/verilog
	  transform/utils


MaseGraph Interface Passes
--------------------------

.. list-table:: A summary of all MaseGraph interface passes 
  :widths: 25 75
  :header-rows: 1

  * - Pass Name
    - Summary
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.load_mase_graph_interface_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.save_mase_graph_interface_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.save_node_meta_param_interface_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.load_node_meta_param_interface_pass`
    - fill me

.. toctree::
	  :maxdepth: 2

	  interface/save_and_load

