import torch.nn as nn
import torch
from torch.fx import symbolic_trace
from chop.passes.graph.analysis.add_metadata.add_common_metadata import graph_iterator_for_mase_ops

# Dummy meta object to simulate the expected structure with an attribute "parameters"
class DummyMaseMeta:
    def __init__(self):
        self.parameters = {"common": {}}

class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(3, 3, 3)

    def forward(self, x):
        return self.deconv(x)

def test_graph_iterator_supports_convtranspose2d():
    model = SampleModel()
    gm = symbolic_trace(model)
    gm.fx_graph = gm.graph             
    gm.model = gm                      
    gm.custom_ops = {"modules": {}}
    gm.model.patched_custom_layers = ()
    gm.modules = dict(gm.named_modules())

    for node in gm.graph.nodes:
        node.meta["mase"] = DummyMaseMeta()

    graph_iterator_for_mase_ops(gm)
