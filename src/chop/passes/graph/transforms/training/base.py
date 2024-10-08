import torch

from chop.ir.common import MASE_TYPE_MAP
from chop.passes.graph.utils import (
    get_mase_op, 
    get_mase_type, 
    get_node_actual_target,
    get_parent_name)


# empty for now
EDITABLE_OPS = []


def compute_sparse_tensor_with_l1_norm(x, sparsity=0.5):
    if not (0 <= sparsity <= 1):
        raise ValueError("Sparsity ratio must be between 0 and 1.")

    x_flattened = x.view(-1)
    keep = int((1.0 - sparsity) * x_flattened.numel())

    if keep == 0:
        return torch.zeros_like(x)
    
    abs_x_flattened = x_flattened.abs()
    value, _ = torch.topk(abs_x_flattened, keep, sorted=False)

    sparse_x = x * (torch.abs(x) >= value[-1]).float()
    return sparse_x

# Inherit from Function
class LinearFunction(torch.autograd.Function):

    @staticmethod
    def forward(inputs, weight, bias, extra_params):
        sparse_w = compute_sparse_tensor_with_l1_norm(
            weight, extra_params["forward_w_sparsity"])
        sparse_x = compute_sparse_tensor_with_l1_norm(
            inputs, extra_params["forward_x_sparsity"])
        output = sparse_x.mm(sparse_w.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        inputs, weight, bias, extra_params = inputs
        ctx.save_for_backward(inputs, weight, bias)
        # critical line for storing extra parameters to be used in backward pass
        for k, v in extra_params.items():
            setattr(ctx, k, v)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        sparse_w = compute_sparse_tensor_with_l1_norm(
            weight, ctx.backward_w_sparsity)
        sparse_grad_y = compute_sparse_tensor_with_l1_norm(
            grad_output, ctx.backward_grad_y_sparsity)
        sparse_x = compute_sparse_tensor_with_l1_norm(
            inputs, ctx.backward_x_sparsity)

        if ctx.needs_input_grad[0]:
            grad_input = sparse_grad_y.mm(sparse_w)
            # grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = sparse_grad_y.t().mm(sparse_x)
            # grad_weight = grad_output.t().mm(inputs)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = sparse_grad_y.sum(0)
            # grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None


# Inherit from nn.Linear Module
class SparseLinear(torch.nn.Linear):
    def __init__(
            self, in_features, out_features, 
            bias=True, device=None, dtype=None, 
            extra_params={
                "forward_w_sparsity": 0.5, 
                "forward_x_sparsity": 0.5,
                "backward_w_sparsity": 0.5,
                "backward_grad_y_sparsity": 0.5,
                "backward_x_sparsity": 0.5}):

        super(SparseLinear, self).__init__(
            in_features=in_features, 
            out_features=out_features, 
            bias=bias,
            device=device, 
            dtype=dtype
        )
        self.extra_params = extra_params
    
    def forward(self, inputs):
        return LinearFunction.apply(inputs, self.weight, self.bias, self.extra_params)




def graph_iterator_by_type(graph, config: dict):
    for node in graph.fx_graph.nodes:
        mase_op = get_mase_op(node)
        # if mase_op not in EDITABLE_OPS:
        #     continue
        node_config = config.get(mase_op, None)

        # node not defined, skip
        # maybe add default support, or print these skipped ones later
        if node_config is None:
            continue

        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            if mase_op == "linear":
                new_module = SparseLinear(
                    ori_module.in_features,
                    ori_module.out_features,
                    bias=ori_module.bias is not None,
                    extra_params=node_config)
                parent_name, name = get_parent_name(node.target)
                setattr(graph.modules[parent_name], name, new_module)
    return graph


def backward_pass_base(graph, pass_args: dict = {}):
    by = pass_args.pop("by")
    match by:
        case "type":
            graph = graph_iterator_by_type(graph, pass_args)
        # case "name":
        #     graph = graph_iterator_by_name(graph, pass_args)
        case _:
            raise ValueError(f'Unsupported quantize "by": {by}')

    # link the model with graph
    graph.model = torch.fx.GraphModule(graph.model, graph.fx_graph)
    return graph, {}
