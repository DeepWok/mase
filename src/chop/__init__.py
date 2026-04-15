try:
    from .ir.graph.mase_graph import MaseGraph
except ImportError:
    MaseGraph = None

try:
    from .ir.onnx.mase_onnx_graph import MaseOnnxGraph
except ImportError:
    MaseOnnxGraph = None

from . import passes

try:
    from .pipelines import AutoPipelineForDistributedInference
except ImportError:
    AutoPipelineForDistributedInference = None
