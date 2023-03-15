from torch.nn import Linear, ModuleList
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaEncoder,
    RobertaIntermediate,
    RobertaLayer,
    RobertaModel,
    RobertaOutput,
    RobertaSelfAttention,
    RobertaSelfOutput,
)

"""
roberta
In [3]: MACs_linear/MACs_all = 100*(1-589.82*1e3/(48.32*1e9))
Out[3]: 99.9987793460265
"""

ignore_modules = [
    RobertaModel,
    RobertaEncoder,
    ModuleList,
    RobertaLayer,
    RobertaAttention,
    RobertaSelfAttention,
    RobertaSelfOutput,
    RobertaIntermediate,
    RobertaOutput,
    Linear,
]


config = dict(
    print_profile=True,
    detailed=True,
    module_depth=-1,
    top_modules=1,
    warm_up=10,
    as_string=True,
    output_file="estimate-sw_reports/roberta_no_linear_layer.txt",
    ignore_modules=ignore_modules,
)
