import torch
import types

from chop.tools import get_logger

from chop.passes.graph.transforms.pruning.load import load_activation_prune_config, load_weight_prune_config
from chop.passes.graph.transforms.pruning.pruning_methods import weight_criteria_map, activation_criteria_map
from chop.passes.graph.transforms.pruning.sparse_parameterization import FakeSparseWeight, FakeStructuredSparseWeight
from chop.passes.graph.transforms.pruning.hwpq import HWPQParameterization

logger = get_logger(__name__)
logger.setLevel("INFO")


def prune_with_a_function(info, fn, sparsity):
    return fn(info, sparsity)


def get_weight_rank_fn(c):
    return weight_criteria_map[c["scope"]][c["granularity"]][c["method"]]


def get_activation_rank_fn(c):
    return activation_criteria_map[c["scope"]][c["granularity"]][c["method"]]


def get_weight_hook(name, info, named_info, w_config: dict):
    """
    get_weight_hook is called for each node 'name' in the FX graph.
    'name' is the node_name (e.g. "encoder.layers.0.attention.q_proj").
    'info' is the entire dictionary of node -> metadata.
    'named_info' is just this node's metadata.
    """
    w_rank_fn = get_weight_rank_fn(w_config)
    value = named_info["value"]
    w_sparsity = named_info["weight_sparsity"]
    register_parameter_name = "weight"

    # Add structured_sparsity flag if using HWPQ
    if w_config["method"] == "hwpq":
        named_info["structured_sparsity"] = w_config.get("structured_sparsity", True)

    if w_config["scope"] == "global":
        param_mask = w_rank_fn(value, info, w_sparsity, node_name=name)
    else:
        param_mask = w_rank_fn(value, named_info, w_sparsity)

    # Use special parameterization for HWPQ to handle both pruning and quantization
    if w_config["method"] == "hwpq":
        parameterization = HWPQParameterization(param_mask)
    else:
        parameterization = FakeSparseWeight(param_mask)
    
    return (register_parameter_name, parameterization)


def get_activation_hook(name, info, named_info, a_config: dict):
    a_rank_fn = get_activation_rank_fn(a_config)
    a_sparsity = named_info["activation_sparsity"]

    def sparsify_input(module, args):
        if len(args) > 1:
            raise ValueError(
                f"{module.__class__.__name__} takes more than 1 argument at inference,"
                " the current 'sparsify_input' pre-forward hook only allows one!"
            )
        x = args[0]

        if a_config["scope"] == "global": # Extra logic required for eventual movement pruning
            mask = a_rank_fn(x, info, a_sparsity, node_name=name) 
        else:
            mask = a_rank_fn(x, named_info, a_sparsity) # Changed from info
        module.activation_mask = mask
        return x * mask
    return ("register_forward_pre_hook", sparsify_input)


def build_pruning_hooks(info, w_config, a_config):
    named_hooks = {}
    for k, v in info.items():
        if v is not None:
            w_info = {
                "module_type": v["module_type"],
                "weight_sparsity": w_config["sparsity"],
                "value": v["weight_value"],
                "shape": v["weight_shape"],
            }
            if "weight_stats" in v.keys():
                w_info["stats"] = v["weight_stats"]

            # for activations
            a_info = {
                "module_type": v["module_type"],
                "activation_sparsity": a_config["sparsity"],
                "value": v["activation_value"],
                "shape": v["activation_shape"],
            }
            if "activation_stats" in v.keys():
                a_info["stats"] = v["activation_stats"]
            named_hooks[k] = {
                "w_hook": get_weight_hook(k, info, w_info, w_config),
                "a_hook": get_activation_hook(k, info, a_info, a_config),
            }
    return named_hooks

def fetch_info(node, module):
    """
    Fetches metadata for the module from the FX node.
    For Conv2d and Linear modules, if the node's software stats are not present,
    it falls back to the module's metadata (which should contain the movement stats
    updated during training).
    """
    if isinstance(module, torch.nn.Conv2d):
        a_value = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["value"]
        a_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]
        w_value = node.meta["mase"].parameters["common"]["args"]["weight"]["value"]
        w_shape = node.meta["mase"].parameters["common"]["args"]["weight"]["shape"]

        out = {
            "module_type": "conv2d",
            "weight_value": w_value,
            "weight_shape": w_shape,
            "activation_value": a_value,
            "activation_shape": a_shape,
        }

        if "args" in node.meta["mase"].parameters["software"]:
            out["activation_stats"] = node.meta["mase"].parameters["software"]["args"]["data_in_0"]["stat"]
            out["weight_stats"] = node.meta["mase"].parameters["software"]["args"]["weight"]["stat"]
        elif hasattr(module, "metadata"):
            for pname, _ in module.named_parameters():
                if pname in module.metadata:
                    out["weight_stats"] = module.metadata[pname]["stats"]
                    break
        return out

    if isinstance(module, torch.nn.Linear):
        a_value = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["value"]
        a_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]
        w_value = node.meta["mase"].parameters["common"]["args"]["weight"]["value"]
        w_shape = node.meta["mase"].parameters["common"]["args"]["weight"]["shape"]
        out = {
            "module_type": "linear",
            "weight_value": w_value,
            "weight_shape": w_shape,
            "activation_value": a_value,
            "activation_shape": a_shape,
        }

        if "args" in node.meta["mase"].parameters["software"]:
            out["activation_stats"] = node.meta["mase"].parameters["software"]["args"]["data_in_0"]["stat"]
            out["weight_stats"] = node.meta["mase"].parameters["software"]["args"]["weight"]["stat"]
        elif hasattr(module, "metadata"):
            for pname, _ in module.named_parameters():
                if pname in module.metadata:
                    out["weight_stats"] = module.metadata[pname]["stats"]
                    break
        return out

    return None


def check_snip_scores(graph, info):
    """Check if SNIP scores are present in model metadata.
    
    Args:
        graph: The MASE graph
        info: The module info dictionary
        
    Returns:
        bool: True if SNIP scores are present, False otherwise
    """
    has_snip_scores = False
    
    for node_target, node_info in info.items():
        if node_info is None:
            continue
            
        # Check if SNIP scores exist in module metadata
        if "weight_stats" in node_info and "snip_scores" in node_info["weight_stats"]:
            has_snip_scores = True
            break
            
    return has_snip_scores


def prepare_snip_scores(graph, dummy_input=None):
    """Compute SNIP scores for the graph.
    
    This method computes SNIP scores for all prunable layers in the graph.
    
    Args:
        graph: The MASE graph
        dummy_input: Optional dictionary of dummy inputs
        
    Returns:
        The graph with SNIP scores computed
    """
    try:
        from chop.passes.graph.transforms.pruning.snip_helper import SNIPCallback
        logger.info("Computing SNIP scores...")
        
        # Use available dummy input
        if dummy_input is None:
            # Try approach 1: Check for common_dummy_in attribute
            if hasattr(graph, "common_dummy_in"):
                dummy_input = graph.common_dummy_in
                logger.info("Using common_dummy_in from graph")
            
            # Try approach 2: Look for input_values in graph's dummy input
            elif hasattr(graph, "dummy_in") and "input_values" in graph.dummy_in:
                dummy_input = graph.dummy_in
                logger.info("Using dummy_in from graph")
            
            # Try approach 3: Extract from placeholder nodes
            else:
                logger.info("Attempting to extract dummy input from placeholder nodes")
                dummy_input = {}
                for node in graph.fx_graph.nodes:
                    if node.op == "placeholder" and hasattr(node, "meta") and "mase" in node.meta:
                        param_name = node.name
                        if "common" in node.meta["mase"].parameters and "args" in node.meta["mase"].parameters["common"]:
                            if param_name in node.meta["mase"].parameters["common"]["args"]:
                                if "value" in node.meta["mase"].parameters["common"]["args"][param_name]:
                                    dummy_input[param_name] = node.meta["mase"].parameters["common"]["args"][param_name]["value"]
                                    logger.info(f"Found placeholder value for {param_name}")
                
                # If still empty, try one more approach with arg names
                if not dummy_input:
                    for node in graph.fx_graph.nodes:
                        if node.op == "placeholder" and hasattr(node, "meta") and "mase" in node.meta:
                            for arg_name, arg_info in node.meta["mase"].parameters["common"]["args"].items():
                                if "value" in arg_info:
                                    dummy_input[arg_name] = arg_info["value"]
                                    logger.info(f"Found value for {arg_name}")

        # Create simple input if all else fails                                    
        if not dummy_input:
            # Create a minimal dummy input as a last resort for most common cases
            logger.info("Creating minimal dummy input (last resort)")
            input_nodes = [n for n in graph.fx_graph.nodes if n.op == "placeholder"]
            if len(input_nodes) > 0:
                # Typical scenarios for common models
                if any(n.name == "input_values" for n in input_nodes):
                    dummy_input = {
                        "input_values": torch.zeros((1, 16000), dtype=torch.float32),
                    }
                    if any(n.name == "attention_mask" for n in input_nodes):
                        dummy_input["attention_mask"] = torch.ones((1, 16000), dtype=torch.long)
                elif any(n.name == "input_ids" for n in input_nodes):
                    dummy_input = {
                        "input_ids": torch.zeros((1, 32), dtype=torch.long),
                    }
                    if any(n.name == "attention_mask" for n in input_nodes):
                        dummy_input["attention_mask"] = torch.ones((1, 32), dtype=torch.long)
                elif any(n.name == "x" for n in input_nodes):
                    dummy_input = {
                        "x": torch.zeros((1, 3, 224, 224), dtype=torch.float32),
                    }
        
        if dummy_input is None or not dummy_input:
            raise ValueError("No dummy input available for SNIP score computation")
        
        logger.info(f"Using dummy input with keys: {list(dummy_input.keys())}")
        
        # Apply SNIP computation directly to the model
        device = next(graph.model.parameters()).device
        
        # Save original forwards
        original_forwards = {}
        
        # Step 1: Save original forwards and add masks
        for name, module in graph.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and hasattr(module, "weight"):
                original_forwards[name] = module.forward
                
                # Create weight_mask parameter
                if not hasattr(module, "weight_mask"):
                    module.weight_mask = torch.nn.Parameter(torch.ones_like(module.weight))
                
                # Override forward method
                if isinstance(module, torch.nn.Conv2d):
                    def new_forward(self, x):
                        return torch.nn.functional.conv2d(
                            x,
                            self.weight.detach() * self.weight_mask,
                            self.bias,
                            self.stride,
                            self.padding,
                            self.dilation,
                            self.groups,
                        )
                    module.forward = types.MethodType(new_forward, module)
                elif isinstance(module, torch.nn.Linear):
                    def new_forward(self, x):
                        return torch.nn.functional.linear(x, self.weight.detach() * self.weight_mask, self.bias)
                    module.forward = types.MethodType(new_forward, module)
        
        # Step 2: Forward-backward pass
        graph.model.zero_grad()
        
        # Move inputs to device
        inputs_on_device = {}
        for k, v in dummy_input.items():
            if isinstance(v, torch.Tensor):
                inputs_on_device[k] = v.to(device)
            else:
                inputs_on_device[k] = v
        
        # Run forward pass
        try:
            logger.info(f"Running forward pass with inputs: {list(inputs_on_device.keys())}")
            output = graph.model(**inputs_on_device)
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            # Try again with each input separately if there was an error
            for k, v in inputs_on_device.items():
                logger.info(f"Trying single input: {k}")
                try:
                    output = graph.model(**{k: v})
                    logger.info(f"Success with just {k}")
                    inputs_on_device = {k: v}
                    break
                except Exception as ee:
                    logger.info(f"Failed with just {k}: {str(ee)}")
            
            # If we still don't have output, raise the original error
            if 'output' not in locals():
                raise e
        
        # Use a simple loss function: mean of all outputs
        if isinstance(output, dict):
            # If output is a dictionary, find a suitable tensor
            if "last_hidden_state" in output:
                loss = output["last_hidden_state"].abs().mean()
                logger.info("Using last_hidden_state for loss")
            elif "logits" in output:
                loss = output["logits"].abs().mean()
                logger.info("Using logits for loss")
            else:
                # Fall back to first tensor
                for k, v in output.items():
                    if isinstance(v, torch.Tensor):
                        loss = v.abs().mean()
                        logger.info(f"Using {k} for loss")
                        break
        else:
            # If output is a tensor
            loss = output.abs().mean()
            logger.info("Using direct output tensor for loss")
            
        # Compute gradients
        loss.backward()
        
        # Step 3: Store gradients as SNIP scores
        for name, module in graph.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)) and hasattr(module, "weight_mask"):
                grad = module.weight_mask.grad
                if grad is not None:
                    if not hasattr(module, "metadata"):
                        module.metadata = {}
                    if "weight" not in module.metadata:
                        module.metadata["weight"] = {}
                    if "stats" not in module.metadata["weight"]:
                        module.metadata["weight"]["stats"] = {}
                    
                    # Store absolute gradient as SNIP score
                    module.metadata["weight"]["stats"]["snip_scores"] = grad.abs().detach().clone()
                    logger.info(f"Module {name}: SNIP score norm = {grad.abs().norm().item()}")
        
        # Step 4: Restore original forwards
        for name, module in graph.model.named_modules():
            if name in original_forwards:
                module.forward = original_forwards[name]
        
        logger.info("SNIP scores computed successfully")
    except ImportError:
        logger.error("Failed to import SNIPCallback - SNIP preprocessing not available")
    except Exception as e:
        logger.error(f"Error computing SNIP scores: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    return graph


def prune_graph_iterator(graph, config: dict):
    # Setup all pruning-related parameters (incl. basic validation)
    w_config = load_weight_prune_config(config["weight"], graph)
    a_config = load_activation_prune_config(config["activation"], graph)

    # First loop: fetch info from each node
    info = {}
    for node in graph.fx_graph.nodes:
        if node.op == "call_module":
            module = graph.modules[node.target]
            meta = fetch_info(node, module)
            info[node.target] = meta
    
    # Check if SNIP method is being used
    if w_config["method"] == "snip" and not check_snip_scores(graph, info):
        logger.info("SNIP method selected but no SNIP scores found. Computing SNIP scores...")
        # Create a dummy input if possible based on common metadata
        dummy_in = None
        if hasattr(graph, "common_dummy_in"):
            dummy_in = graph.common_dummy_in
        graph = prepare_snip_scores(graph, dummy_in)
        
        # Refresh info with new SNIP scores
        info = {}
        for node in graph.fx_graph.nodes:
            if node.op == "call_module":
                module = graph.modules[node.target]
                meta = fetch_info(node, module)
                info[node.target] = meta

    # Build hooks from the info dictionary
    hooks = build_pruning_hooks(info, w_config, a_config)

    # Second loop: apply the hooks
    for node in graph.fx_graph.nodes:
        if node.op == "call_module":
            name = node.target
            if name in hooks.keys():
                logger.info(f"Pruning module: {node.name}")
                node_hooks = hooks[name]
                if node_hooks["w_hook"] is not None:
                    register_name, parameterization = node_hooks["w_hook"]
                    torch.nn.utils.parametrize.register_parametrization(
                        graph.modules[node.target], register_name, parameterization
                    )
                if node_hooks["a_hook"] is not None:
                    register_fn, hook_fn = node_hooks["a_hook"]
                    getattr(graph.modules[node.target], register_fn)(hook_fn)
    return graph


def prune_transform_pass(graph, pass_args: dict = {}):
    """
    Apply pruning transformation to the given graph.
    This is achieved by adding a register_parametrization hook to weights
    and a register_pre_forward hook to activations.
    
    :param graph: The input graph to be pruned.
    :param pass_args: Optional arguments for the pruning transformation.
    :return: The pruned graph and an empty dictionary.
    """
    # Check for method-specific preprocessing requirements
    if "weight" in pass_args and "method" in pass_args["weight"]:
        method = pass_args["weight"]["method"]
        
        # Save dummy input for later use
        if hasattr(graph, "common_dummy_in"):
            dummy_in = graph.common_dummy_in
        
    graph = prune_graph_iterator(graph, pass_args)
    return graph, {}