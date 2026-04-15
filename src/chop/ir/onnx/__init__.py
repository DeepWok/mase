try:
    from .mase_onnx_graph import MaseOnnxGraph
except ImportError:
    MaseOnnxGraph = None

from .utils import ONNX_TO_TORCH_DTYPE, ONNX_OP_MAPPING
