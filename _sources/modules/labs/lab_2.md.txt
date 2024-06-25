<!-- # Lab 1 for Advanced Deep Learning Systems (ADLS ELEC70109/EE9-AML3-10/EE9-AO25) -->

<br />
<div align="center">
  <a href="https://deepwok.github.io/">
    <img src="../imgs/deepwok.png" alt="Logo" width="160" height="160">
  </a>

  <h1 align="center">Lab 2 for Advanced Deep Learning Systems (ADLS)</h1>

  <p align="center">
    ELEC70109/EE9-AML3-10/EE9-AO25
    <br />
  Written by
    <a href="https://aaron-zhao123.github.io/">Aaron Zhao </a>
  </p>
</div>

# General introduction

In this lab, you will learn how to use analysis and transform pass system in the software stack of MASE.

There are in total 7 tasks you would need to finish, and also 1 optional task.

# Turning you network to a graph

One specific feature of MASE is its capability to transform DL models to a computation graph using the [torch.fx](<https://pytorch.org/docs/stable/fx.html>) framework.

In this section, we will look at how to transform a model to a `MASEGraph`, starting from the [Notebook](./lab2_transform_no_CLI.ipynb).

Once you finished the notebook, consider the following problems:

1. Explain the functionality of `report_graph_analysis_pass` and its printed jargons such as `placeholder`, `get_attr` ... You might find the doc of [torch.fx](https://pytorch.org/docs/stable/fx.html) useful.

2. What are the functionalities of `profile_statistics_analysis_pass` and `report_node_meta_param_analysis_pass` respectively?

## MASE OPs and MASE Types

MASE is designed to be a very high-level intermediate representation (IR), this is very different from the classic [LLVM IR](https://llvm.org/docs/LangRef.html) that you might be familiar with.

The following MASE Types are available:

- `module_related_func`: MASE module_realted_func includes functions under `torch.nn.functional` and the `torch.nn.Module` that wrapps them. For example, `torch.nn.functional.relu` and `torch.nn.ReLU` are both MASE module_related_func.
- `module`: a MASE module is a subclass of `torch.nn.Module` that does not have corresponding `torch.nn.functional` counterpart. For example, `torch.nn.BatchNorm2D` is a MASE module because `torch.nn.functional.batch_norm_2d` does not exist.
- `builtin_func`: MASE builtin_func includes functions under `torch` that are not `torch.nn.functional` and `torch.nn.Module`. For example, `torch.cat` and `torch.bmm` are both MASE builtin_func.
- `placeholder`: a MASE placeholder is the input node of a MASEGraph, i.e., a proxy of input tensor to the network. This type inherits from torch.fx.
- `get_attr`: a MASE get_attr is a node that represents the attribute of a MASE module. This type inherits from torch.fx. An example is `self.scale * x` in a `forward` function where `self.scale` is user-defined `torch.nn.Parameter` and `x` is an intermediate tensor.
- `output`: a MASE output is the output node of a MASEGraph, i.e., a proxy of output tensor to the network. This type also inherits from torch.fx.

You may find more clue in the [this file](../../machop/chop/passes/common.py).

## A deeper dive into the quantisation transform

3. Explain why only 1 OP is changed after the `quantize_transform_pass` .

4. Write some code to traverse both `mg` and `ori_mg`, check and comment on the nodes in these two graphs. You might find the source code for the implementation of `summarize_quantization_analysis_pass` useful.

5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the `pass_args` for your custom network might be different if you have used more than the `Linear` layer in your network.

6. Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the [Quantized Layers](../../machop/chop/passes/transforms/quantize/quantized_modules/linear.py) .

## The command line interface

The same flow can also be executed on the command line throw the `transform` action.

```bash
# make sure you have the same printout
pwd
# it should show
# your_dir/mase-tools/machop

# enter the following command
./ch transform --config configs/examples/jsc_toy_by_type.toml --task cls --cpu=0
```

7. Load your own pre-trained JSC network, and perform perform the quantisation using the command line interface.

## Optional Task: Write your own pass

Many examples of existing passes are in the [source code](../..//machop/chop/passes/__init__.py), the [test files](../../machop/test/passes) for these passes also contain useful information on helping you to understand how these passes are used.

Implement a pass to count the number of FLOPs (floating-point operations) and BitOPs (bit-wise operations).
