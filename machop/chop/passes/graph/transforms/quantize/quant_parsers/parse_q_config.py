import copy
from .q_op_entries import FIXED_OP_ENTRIES

"""
MASE_OP_TO_ENTRIES = {
    <op_name>: {
        "required": (...),
        "optional": (...)
    }
}
"""

def get_q_op_entries(q_name: str, mase_op: str):
    match q_name:
        case "fixed":
            op_entries = FIXED_OP_ENTRIES
        case _:
            raise ValueError(f"Unknown quantization arithmetic name: {q_name}")

    if mase_op not in op_entries:
        raise ValueError(f"Unknown MASE operation name: {mase_op} for quantization arithmetic: {q_name}")

    return op_entries[mase_op]


def parse_node_q_config(q_config: dict, mase_op: str):
    q_op_entries = get_q_op_entries(q_config["name"], mase_op)

    required_keys = q_op_entries["required"]
    optional_keys = q_op_entries["optional"]

    parsed_q_config = {}
    for k in required_keys:
        assert k in q_config, f"Required key {k} not found in q_config: {q_config}"
        parsed_q_config[k] = copy.deepcopy(q_config[k])

    for k in optional_keys:
        if k in q_config:
            parsed_q_config[k] = copy.deepcopy(q_config[k])

    return parsed_q_config
