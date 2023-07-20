#!/usr/bin/env python3
# This example add metadata to the toy custom function model for testing ops
import sys
import logging
import os

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "..",
        "machop",
    )
)

from chop.models import get_resnet18
from chop.passes.graph.mase_graph import MaseGraph
from chop.tools.get_input import InputGenerator
from chop.passes import (
    add_common_metadata_analysis_pass,
    add_mase_ops_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.passes.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.tools.logger import getLogger
from chop.dataset import MyDataModule, get_dataset_info

logger = getLogger("chop")
logger.setLevel(logging.DEBUG)


def main():
    # commit: 8ae425b344e1060e7092a7dd41623377ded33167
    # batch-size = 1 will trigger the bug in add_common_metadata_analysis_pass
    batch_size = 2

    datamodule = MyDataModule(
        model_name="toy_custom_fn",
        batch_size=batch_size,
        dataset_name="cifar10",
        workers=os.cpu_count(),
        tokenizer=None,
        max_token_len=None,
    )
    datamodule.prepare_data()
    datamodule.setup()

    input_generator = InputGenerator(
        datamodule=datamodule,
        task="cls",
        is_nlp_model=False,
        which_dataloader="train",
    )

    info = get_dataset_info("cifar10")
    model = get_resnet18(info, pretrained=False)

    dummy_in = {"x": next(iter(datamodule.train_dataloader()))[0]}

    mg = MaseGraph(model=model)
    mg = init_metadata_analysis_pass(mg, None)
    mg = add_mase_ops_analysis_pass(mg, None)
    mg = add_common_metadata_analysis_pass(mg, dummy_in)
    mg = add_software_metadata_analysis_pass(mg, None)

    pass_arg = {
        "by": "type",
        "target_weight_nodes": [
            "conv2d",
        ],
        "target_act_nodes": [
            "relu",
        ],
        "weight_stats": {
            "variance_precise": {"device": "cpu", "dims": "all"},
        },
        "act_stats": {
            "variance_precise": {"device": "cpu", "dims": "all"},
        },
        "input_generator": input_generator,
        "num_samples": 1,
    }
    mg = profile_statistics_analysis_pass(mg, pass_arg)
    mg = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})


if __name__ == "__main__":
    main()
