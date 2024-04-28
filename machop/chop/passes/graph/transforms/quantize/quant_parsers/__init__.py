# from .parse_quant_config import parse_node_q_config
from .parse_q_config import parse_node_q_config

# from .update_node_meta import relink_node_meta, update_quant_meta_param
from .update_node_meta import (
    relink_node_meta,
    update_q_meta_param,
    infer_result_dtype_and_precision,
)
