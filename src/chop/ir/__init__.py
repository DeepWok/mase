try:
    from .graph.mase_graph import MaseGraph, MaseTracer
except ImportError:
    MaseGraph = None
    MaseTracer = None

try:
    from .onnx.mase_onnx_graph import MaseOnnxGraph
except ImportError:
    MaseOnnxGraph = None
