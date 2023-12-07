Chop Passes
============================




MaseGraph Analysis Passes
-----------

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
    -  fill
  * - :py:meth:`~chop.passes.graph.analysis.verify.verify.verify_metadata_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.graph.analysis.verify.verify.verify_common_metadata_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.graph.analysis.verify.verify.verify_software_metadata_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.graph.analysis.verify.verify.verify_hardware_metadata_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.graph.analysis.total_bits_estimator.total_bits_mg.total_bits_mg_analysis_pass`
    -  Perform total bits analysis on the given graph.


.. toctree::
	  :maxdepth: 2

	  analysis/add_metadata
	  analysis/init_metadata
	  analysis/report
	  analysis/statistical_profiler
	  analysis/verify
	  analysis/total_bits_estimator


MaseGraph Transform Passes
-----------

.. list-table:: A summary of all MaseGraph transform passes 
  :widths: 25 75
  :header-rows: 1

  * - Pass Name
    - Summary
  * - :py:meth:`~chop.passes.graph.transforms.pruning.prune.prune_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.pruning.prune.prune_unwrap_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.quantize.quantize.quantize_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.quantize.quantize.summarize_quantization_analysis_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.utils.conv_bn_fusion_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.utils.logicnets_fusion_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.utils.onnx_annotate_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_top.emit_verilog_top_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_top.emit_verilog_tb_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_top.emit_bram_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_top.emit_mlir_hls_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.transforms.verilog.emit_top.emit_internal_rtl_transform_pass`
    - fill me


MaseGraph Interface Passes
-----------

.. list-table:: A summary of all MaseGraph interface passes 
  :widths: 25 75
  :header-rows: 1

  * - Pass Name
    - Summary
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.load_mase_graph_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.save_mase_graph_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.save_node_meta_param_transform_pass`
    - fill me
  * - :py:meth:`~chop.passes.graph.interface.save_and_load.load_node_meta_param_transform_pass`
    - fill me


.. toctree::
	  :maxdepth: 2

	  transform/interface
	  transform/pruning
	  transform/quantize
	  transform/verilog
	  transform/utils

