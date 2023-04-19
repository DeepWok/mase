import os
from logging import getLogger
from typing import Callable, Dict, List

import regex as re
import toml

from ...graph.mase_tracer import (
    is_leaf_module_to_trace,
    mark_as_user_custom_leaf_module,
)
from ...modify.modifier import (
    register_cls_name_to_cls_for_modify_sw,
    register_modified_cls_for_modify_sw,
    register_original_to_name_to_modified_for_modify_sw,
)
from .opt_patched.custom_modules import (
    OPT_MODIFIED_MODULE_CLASSES,
    OPT_MODULE_CLS_MAP,
    OPT_MODULE_CLS_NAME_TO_MODULE_CLS,
    OPT_MODULE_CLS_TO_MODULE_CLS_NAME,
    opt_create_new_custom_module,
)

logger = getLogger(__name__)


def get_leaf_module_classes_to_trace(
    config: Dict,
    custom_module_cls_name_to_module_cls: Dict[str, type],
) -> List:
    original_module_classes_to_trace = []

    if "module_classes_to_modify" in config:
        for cls_name, sub_config in config["module_classes_to_modify"].items():
            if cls_name in custom_module_cls_name_to_module_cls:
                if sub_config["name"] == "NA":
                    continue
                original_cls = custom_module_cls_name_to_module_cls[cls_name]
                original_module_classes_to_trace.append(original_cls)
    if "module_nodes_to_modify" in config:
        for node_name, sub_config in config["module_nodes_to_modify"].items():
            if "original_module_cls" in sub_config:
                cls_name = sub_config["original_module_cls"]
                if (
                    sub_config["original_module_cls"]
                    not in custom_module_cls_name_to_module_cls
                ):
                    logger.warning(
                        f"`Unsupported module cls `{cls_name}` (module node config of Node {node_name})"
                    )
                    continue
                if sub_config["name"] == "NA":
                    continue
                original_cls = custom_module_cls_name_to_module_cls[cls_name]
                if (
                    original_cls in original_module_classes_to_trace
                    or is_leaf_module_to_trace(original_cls)
                ):
                    continue
                else:
                    original_module_classes_to_trace.append(original_cls)
    return original_module_classes_to_trace


def dummy_create_new_module_fn(original_module, config: Dict):
    raise NotImplementedError(
        f"Module {original_module} is not a built-in supported module class to modify"
    )


def get_custom_modify_sw_kwargs(model_name, config_path: str):
    """
    - Parse modify-sw config and register custom leaf modules
    - Return custom_module_mapping to be registered and custom_module_create_fn
    """
    assert os.path.isfile(config_path) and config_path.endswith(".toml")
    config = toml.load(config_path)
    if re.match(r"(facebook\/opt)-([1-9]\d*(\.\d+)?)[m|b]@patched", model_name):
        custom_module_cls_to_trace = get_leaf_module_classes_to_trace(
            config,
            custom_module_cls_name_to_module_cls=OPT_MODULE_CLS_NAME_TO_MODULE_CLS,
        )
        custom_module_cls_map = OPT_MODULE_CLS_MAP
        custom_module_cls_name_to_cls = OPT_MODULE_CLS_NAME_TO_MODULE_CLS
        custom_modified_module_cls = OPT_MODIFIED_MODULE_CLASSES
        custom_module_create_fn = opt_create_new_custom_module
    else:
        logger.debug(f"No custom leaf module registered")
        return dummy_create_new_module_fn

    # for cls in module_classes_to_trace:
    #     mark_as_user_custom_leaf_module(cls)
    # register_original_to_name_to_modified_for_modify_sw(module_cls_map_to_register)
    # register_cls_name_to_cls_for_modify_sw(module_cls_name_to_module_cls_to_register)
    # register_modified_cls_for_modify_sw(OPT_MODIFIED_MODULE_CLASSES)
    # if len(module_classes_to_trace) == 0:
    #     logger.info("No custom leaf module class is additionally registered")
    # else:
    #     logger.info(
    #         "{} custom leaf module classes are registered".format(
    #             len(module_classes_to_trace)
    #         )
    #     )
    return dict(
        custom_module_cls_to_trace=custom_module_cls_to_trace,
        custom_module_cls_map=custom_module_cls_map,
        custom_module_cls_name_to_cls=custom_module_cls_name_to_cls,
        custom_module_create_fn=custom_module_create_fn,
        custom_modified_module_cls=custom_modified_module_cls,
    )
