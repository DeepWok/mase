# MaseGraph and MaseGraph Passes

## MASE OPs and MASE Types

MASE is designed to be an intermediate representation (IR), this is very different from the classic [LLVM IR](https://llvm.org/docs/LangRef.html) that you might be familiar with.

The following MASE Types are available:

- `module_related_func`: MASE module_realted_func includes functions under `torch.nn.functional` and the `torch.nn.Module` that wrapps them. For example, `torch.nn.functional.relu` and `torch.nn.ReLU` are both MASE module_related_func.
- `module`: a MASE module is a subclass of `torch.nn.Module` that does not have corresponding `torch.nn.functional` counterpart. For example, `torch.nn.BatchNorm2D` is a MASE module because `torch.nn.functional.batch_norm_2d` does not exist.
- `builtin_func`: MASE builtin_func includes functions under `torch` that are not `torch.nn.functional` and `torch.nn.Module`. For example, `torch.cat` and `torch.bmm` are both MASE builtin_func.
- `placeholder`: a MASE placeholder is the input node of a MASEGraph, i.e., a proxy of input tensor to the network. This type inherits from torch.fx.
- `get_attr`: a MASE get_attr is a node that represents the attribute of a MASE module. This type inherits from torch.fx. An example is `self.scale * x` in a `forward` function where `self.scale` is user-defined `torch.nn.Parameter` and `x` is an intermediate tensor.
- `output`: a MASE output is the output node of a MASEGraph, i.e., a proxy of output tensor to the network. This type also inherits from torch.fx.

You may find more clue in the [this source code file](https://github.com/DeepWok/mase/blob/main/src/chop/passes/__init__.py).

## MASEGraph Passes

These passes can only be applied when you transform the model to a MASEGraph:

```python
mg = MaseGraph(model=model)

# Three different analysis passes
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)
```

You will see the MASE OPs in the graph. For example, `module_related_func` and `module` are the most common MASE OPs in the graph.