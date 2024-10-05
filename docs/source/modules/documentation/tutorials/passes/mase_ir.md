# The MASE IR System

The aim of the MASE IR system is to translate the fx graph representation to another intermediate representation (IR) that is more suitable for the MASE system. The MASE IR system is designed to be an intermediate representation (IR), this is very different from the classic [LLVM IR](https://llvm.org/docs/LangRef.html) that you might be familiar with.

The reason for such a translation is that the fx graph representation is inherently represent objects following the Pytorch API. Certain complexities, such as dual definition as Module and Functional object for the same operation, are not ideal for a concise description.

It is also worth noting that the MASE IR system is designed as annotated metadata on top of the existing fx graph. This provides dual-use of the fx graph representation and the MASE IR system. 


## Inspecting the fx graph

The following code snippet shows you what are the components in a fx graph:

```python
from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    report_graph_analysis_pass,
    report_node_type_analysis_pass,
)

# --------------------------------------------------
#   Model specifications
# --------------------------------------------------
class MLP(torch.nn.Module):
    """
    Toy quantized FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 28 * 28)
        self.fc2 = nn.Linear(28 * 28, 28 * 28 * 4)
        self.fc3 = nn.Linear(28 * 28 * 4, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        # w = torch.randn((4, 28 * 28))
        # x = torch.nn.functional.relu(nn.functional.linear(x, w))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

mlp = MLP()
mg = MaseGraph(model=mlp)

# Provide a dummy input for the graph so it can use for tracing
batch_size = 1
x = torch.randn((batch_size, 28, 28))
dummy_in = {"x": x}

mg, _ = report_graph_analysis_pass(mg, {})
```

The above code should give you a printout of the fx graph:

```bash
Graph Analysis Report

===================== Graph Summary =====================

    opcode         name     target                                                   args        kwargs
-------------  -------  -------------------------------------------------------  ----------  -------------------------------
placeholder    x        x                                                        ()          {}
call_function  flatten  <built-in method flatten of type object at 0x10899b790>  (x,)        {'start_dim': 1, 'end_dim': -1}
call_module    fc1      fc1                                                      (flatten,)  {}
call_function  relu     <function relu at 0x109f74b80>                           (fc1,)      {'inplace': False}
call_module    fc2      fc2                                                      (relu,)     {}
call_function  relu_1   <function relu at 0x109f74b80>                           (fc2,)      {'inplace': False}
call_module    fc3      fc3                                                      (relu_1,)   {}
output         output   output                                                   (fc3,)      {}
```

## Inspecting the MASE IR system

We can then try to investigate how the fx graph is augmented with the MASE IR system:

```python
mg, _ = init_metadata_analysis_pass(mg, {})
mg, _ = add_common_metadata_analysis_pass(
    mg, {"dummy_in": dummy_in, "add_value": False}
)
mg, _ = report_node_type_analysis_pass(mg, {})

```

This will give you the following printout:

```bash
INFO
Node name    Fx Node op     Mase type            Mase op      Value type
-----------  -------------  -------------------  -----------  ------------
x            placeholder    placeholder          placeholder  NA
flatten      call_function  implicit_func        flatten      float
fc1          call_module    module_related_func  linear       float
relu         call_function  module_related_func  relu         float
fc2          call_module    module_related_func  linear       float
relu_1       call_function  module_related_func  relu         float
fc3          call_module    module_related_func  linear       float
output       output         output               output       NA
```

Clearly, you can now see each graph node is annotated with the MASE IR with MASE type, MASE op, and value type. This is the MASE IR system in action.
The above code also illustrates that MASE IR is achieved through two passes: `init_metadata_analysis_pass` and `add_common_metadata_analysis_pass`. These two passes together inject information (MASE IR information in this case) into the metadata entry of each node in the graph. The `report_node_type_analysis_pass` is then used to print out the MASE IR information.

## MASE IR Details

MASE IR classifies ops into seen types.

- "module": ops that are related to torch modules only
- "module_related_func": ops that are related to torch modules and functions
- "builtin_func": ops that are related to built-in functions
- "implicit_func": ops that are related to implicit functions
- "placeholder": ops that are related to placeholders, normally for data input
- "get_attr": ops that are related to getting attributes from the model
- "output": ops that are related to the output of the model

Full definitions and ops related to each type can be found in `src/chop/ir/common.py`.