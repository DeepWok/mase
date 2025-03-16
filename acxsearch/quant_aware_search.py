
from a_cx_mxint_quant.module_level_tranform import vit_module_level_quantize
from utils import acc_cal

import timm

from utils import init_dataset
from chop.tools import get_logger
from chop.tools.logger import set_logging_verbosity

from a_cx_mxint_quant import DEIT_TINY_IMAGENET_ACC_100ITER
import toml

def quant_evaluation(model, quant_config, datamodule, max_iteration=100):
    # Set to evaluation mode if needed
    vit_module_level_quantize(model, quant_config)
    acc = acc_cal(model, datamodule.test_dataloader(), max_iteration=max_iteration)
    return acc


def quant_evaluation(model, quant_config, datamodule, max_iteration=100):
    # Set to evaluation mode if needed
    vit_module_level_quantize(model, quant_config)
    acc = acc_cal(model, datamodule.test_dataloader(), max_iteration=max_iteration)
    return acc
def iterative_search(checkpoint, target_op, search_args):
    # load model and datamodule
    model = timm.create_model(checkpoint, pretrained=True)
    datamodule = init_dataset("imagenet", 32, checkpoint)
    # loop through search_args
    quant_acc = DEIT_TINY_IMAGENET_ACC_100ITER
    result_dict = {}
    for arg, search_range in search_args.items():
        logger.info(f"start {arg} search")
        acc_list = []
        for value in search_range:
            quant_config[target_op]["config"].update({arg: value})
            logger.debug(f"quant_config: {quant_config}")

            acc = quant_evaluation(model, quant_config, datamodule, max_iteration=100)
            acc_list.append(acc)
            # Store intermediate result in result_dict
            result_key = f"{target_op}_{arg}_{value}"
            result_dict[result_key] = acc
            # Save results to toml file after each update
            with open(f"{checkpoint}_search_results.toml", "w") as f:
                toml.dump(result_dict, f)
            if acc > quant_acc - 0.001:
                logger.info(f"acc: {acc}, quant_acc: {quant_acc}, {arg}: {value}")
                break

        best_acc = max(acc_list)
        best_param = search_range[acc_list.index(best_acc)]
        quant_config[target_op]["config"].update({arg: best_param})
        logger.info(f"{arg}: {best_param}, acc: {best_acc}")

    final_acc = acc_cal(model, datamodule.test_dataloader())
    logger.info(f"quant_config: {quant_config}")
    logger.info(f"original_acc: {DEIT_TINY_IMAGENET_ACC}, final_acc: {final_acc}")
    return quant_config