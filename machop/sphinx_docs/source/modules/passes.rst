Chop Passes
============================


Mase Graph
-----------

Mase MetaData
-------------


Analysis Passes
-----------

.. list-table:: A summary of all analysis passes 
  :widths: 25 75
  :header-rows: 1

  * - Pass Name
    - Summary
  * - :py:meth:`~chop.passes.analysis.add_metadata.add_common_metadata.add_common_metadata_analysis_pass`
    - Add metadata used for both software and hardware
  * - :py:meth:`~chop.passes.analysis.add_metadata.add_software_metadata.add_software_metadata_analysis_pass`
    - Add hardware-specific metadata
  * - :py:meth:`~chop.passes.analysis.add_metadata.add_hardware_metadata.add_hardware_metadata_analysis_pass`
    - Add software-specific metadata
  * - :py:meth:`~chop.passes.analysis.init_metadata.init_metadata_analysis_pass`
    - Initialize each node with the MaseMetadata 
  * - :py:meth:`~chop.passes.analysis.report.report_graph.report_graph_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.analysis.report.report_node.report_node_hardware_type_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.analysis.report.report_node.report_node_meta_param_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.analysis.report.report_node.report_node_shape_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.analysis.report.report_node.report_node_type_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.analysis.statistical_profiler.profile_statistics.profile_statistics_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.analysis.verify.verify.verify_metadata_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.analysis.verify.verify.verify_common_metadata_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.analysis.verify.verify.verify_software_metadata_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.analysis.verify.verify.verify_hardware_metadata_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.analysis.total_bits_estimator.total_bits_mg.total_bits_mg_analysis_pass`
    -  fill
  * - :py:meth:`~chop.passes.analysis.total_bits_estimator.total_bits_module.total_bits_module_analysis_pass`
    -  fill

.. toctree::
	  :maxdepth: 2

	  analysis/add_metadata
	  analysis/init_metadata
	  analysis/report
	  analysis/statistical_profiler
	  analysis/verify
	  analysis/fpgaconvnet
	  analysis/total_bits_estimator


Transform Passes
-----------

.. list-table:: A summary of all transform passes 
  :widths: 25 75
  :header-rows: 1

  * - Pass Name
    - Summary
  * - :py:meth:`~chop.passes.transform.add_metadata.add_common_metadata.add_common_metadata_analysis_pass`
    - Add metadata used for both software and hardware

.. toctree::
	  :maxdepth: 2


.. toctree::
    :maxdepth: 2
    :caption: Package Reference

    ../modules/actions
