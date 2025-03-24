
from a_cx_mxint_quant.module_level_tranform import vit_module_level_quantize
from utils import acc_cal

import timm
import json
from utils import init_dataset
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity

from a_cx_mxint_quant import DEIT_TINY_IMAGENET_ACC, DEIT_TINY_IMAGENET_ACC_100ITER

logger = get_logger(__name__)
set_logging_verbosity("info")

def quant_evaluation(model, quant_config, datamodule, max_iteration=100):
    # Set to evaluation mode if needed
    vit_module_level_quantize(model, quant_config)
    acc = acc_cal(model, datamodule.test_dataloader(), max_iteration=max_iteration)
    return acc

def iterative_search(checkpoint, target_op, search_args, quant_config):
    # load model and datamodule
    model = timm.create_model(checkpoint, pretrained=True)
    datamodule = init_dataset("imagenet", 32, checkpoint)
    # loop through search_args
    quant_acc = DEIT_TINY_IMAGENET_ACC_100ITER
    # Check if search results file already exists
    import os
    result_file = f"{checkpoint}_search_results.json"
    result_dict = {}
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            result_dict = json.load(f)
        logger.info(f"Loaded existing search results from {result_file}")


    for arg, search_range in search_args.items():
        result_dict[f"{target_op}_{arg}"] = {}
        logger.info(f"start {arg} search")
        acc_list = []
        for value in search_range:
            quant_config[target_op]["config"].update({arg: value})
            logger.debug(f"quant_config: {quant_config}")

            acc = quant_evaluation(model, quant_config, datamodule, max_iteration=100)
            acc_list.append((value, acc))
            if (acc > quant_acc - 0.001) and (acc < quant_acc + 0.001):
                logger.info(f"acc: {acc}, quant_acc: {quant_acc}, {arg}: {value}")
                break

        best_acc = max(acc_list)
        best_param = search_range[acc_list.index(best_acc)]
        quant_config[target_op]["config"].update({arg: best_param})
        logger.info(f"{arg}: {best_param}, acc: {best_acc}")

        result_dict[f"{target_op}_{arg}"] = {
            "search_log": acc_list,
            "best_result": {
                "best_acc": best_acc,
                "best_param": best_param,
            }
        }
        with open(result_file, "w") as f:
            json.dump(result_dict, f, indent=4)

    final_acc = acc_cal(model, datamodule.test_dataloader())
    result_dict[f"{target_op}_final"] = {
        "best_config": quant_config,
        "best_result": {
            "best_acc": final_acc,
        }
    }
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=4)
    logger.info(f"quant_config: {quant_config}")
    logger.info(f"original_acc: {DEIT_TINY_IMAGENET_ACC}, final_acc: {final_acc}")
    return quant_config