# #!/usr/bin/env python3
# # This example add metadata to the toy custom function model for testing ops
# import logging
# import os
# import sys
#
# os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"
#
# sys.path.append(
#     os.path.join(
#         os.path.dirname(os.path.realpath(__file__)),
#         "..",
#         "..",
#         "..",
#         "..",
#         "..",
#         "..",
#         "machop",
#     )
# )
#
# from chop.dataset import MaseDataModule, get_dataset_info
# from chop.models import get_model, get_model_info
# from chop.passes.graph import (
#     add_common_metadata_analysis_pass,
#     add_software_metadata_analysis_pass,
#     init_metadata_analysis_pass,
# )
# from chop.passes.graph.analysis import (
#     profile_statistics_analysis_pass,
#     report_node_meta_param_analysis_pass,
# )
# from chop.tools.logger import set_logging_verbosity
# from chop.ir.graph.mase_graph import MaseGraph
# from chop.tools.get_input import InputGenerator
#
# set_logging_verbosity("debug")
#
#
# def test_statistic_profiler():
#     # commit: 8ae425b344e1060e7092a7dd41623377ded33167
#     # batch-size = 1 will trigger the bug in add_common_metadata_analysis_pass
#     batch_size = 2
#
#     model_info = get_model_info("toy_custom_fn")
#
#     datamodule = MaseDataModule(
#         model_name="toy_custom_fn",
#         batch_size=batch_size,
#         name="cifar10",
#         num_workers=os.cpu_count(),
#         tokenizer=None,
#         max_token_len=None,
#     )
#     datamodule.prepare_data()
#     datamodule.setup()
#
#     input_generator = InputGenerator(
#         model_info=model_info,
#         data_module=datamodule,
#         task="cls",
#         which_dataloader="train",
#     )
#
#     dataset_info = get_dataset_info("cifar10")
#     model = get_model(
#         "resnet18", task="cls", dataset_info=dataset_info, pretrained=False
#     )
#
#     dummy_in = {"x": next(iter(datamodule.train_dataloader()))[0]}
#
#     mg = MaseGraph(model=model)
#     mg = init_metadata_analysis_pass(mg, None)
#     mg = add_common_metadata_analysis_pass(mg, dummy_in)
#     mg = add_software_metadata_analysis_pass(mg, None)
#
#     pass_arg = {
#         "by": "type",
#         "target_weight_nodes": [
#             "conv2d",
#         ],
#         "target_activation_nodes": [
#             "relu",
#         ],
#         "weight_statistics": {
#             "variance_precise": {"device": "cpu", "dims": "all"},
#         },
#         "activation_statistics": {
#             "variance_precise": {"device": "cpu", "dims": "all"},
#         },
#         "input_generator": input_generator,
#         "num_samples": 1,
#     }
#     mg = profile_statistics_analysis_pass(mg, pass_arg)
#     mg = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})
