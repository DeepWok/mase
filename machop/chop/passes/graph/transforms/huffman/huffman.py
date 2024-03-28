

def huffman_transform_pass(
        pl_model, 
        cf_args, 
        model_info,
        data_module,
        task,
        accelerator,
        pass_config
        ):
    import torch
    import pickle

    from chop.passes.graph import PASSES
    from chop.tools.get_input import get_dummy_input
    from chop.passes.graph.utils import get_mase_op, get_node_actual_target

    from chop.passes.graph.analysis import (
        add_common_metadata_analysis_pass,
        add_software_metadata_analysis_pass,
        init_metadata_analysis_pass,
    )

    from chop.ir.graph import MaseGraph
    from chop.tools.get_input import get_dummy_input
    from chop.ir.graph.mase_graph import MaseGraph
    from chop.tools.get_input import InputGenerator, get_cf_args, get_dummy_input
    from chop.tools.utils import parse_accelerator, to_numpy_if_tensor
    from chop.passes.graph.transforms import metadata_value_type_cast_transform_pass
    import pdb

    from collections import Counter, defaultdict
    from heapq import heapify, heappush, heappop

    import gc
    gc.collect()
    ###############################
    # post-train-quantization & huffman coding
    ###############################
    # post-train-quantization
    new_graph = MaseGraph(model=pl_model, cf_args=cf_args)  # pl_model.state_dict()
    new_graph, _ = init_metadata_analysis_pass(new_graph, pass_args=None)

    dummy_in = get_dummy_input(model_info=model_info, data_module=data_module, task=task, device=accelerator)
    new_graph, _ = add_common_metadata_analysis_pass(new_graph, pass_args={"dummy_in": dummy_in})
    new_graph, _ = add_software_metadata_analysis_pass(new_graph, pass_args=None)

    pass_config = pass_config['quantize']
    pass_config['by'] = "type"
    gc.collect()
    new_graph, _ = metadata_value_type_cast_transform_pass(new_graph, pass_args={"fn": to_numpy_if_tensor})
    new_graph, _ = PASSES["quantize"](new_graph, pass_args=pass_config)

    for node in new_graph.nodes:
        if isinstance(get_node_actual_target(node), torch.nn.modules.Conv2d): 
            if 'mase' in node.meta:
                quantized_weight = get_node_actual_target(node).w_quantizer(get_node_actual_target(node).weight)
                new_graph.model.state_dict()['.'.join(('.'.join(node.name.rsplit('_', 1))).split('_', 1)) + ".weight"].copy_(quantized_weight)
                print(f"There is quantization at {node.name}, mase_op: {get_mase_op(node)}")

    '''
    Note: As there is at current no support in PyTorch for float4 or float 8 operations, the lowest is float16 (or int8 which is not suitable for us)
    Even if we make quantization to 8 digits, the dtype is still float32
    Therefore, the actual model size does not actually decrease.
    We would investigate on hardward part further to realize this.
    '''
    torch.save(new_graph.model.state_dict(), "chop/post_train_pruned_model.ckpt")


    # start huffman coding

    state_dict = new_graph.model.state_dict()
    huffman_size_bits = 0
    layer_huffman_info = defaultdict(dict)

    for layer_name, weight_tensor in state_dict.items():
        if "weight" in layer_name and len(weight_tensor.size()) == 4:  # conv2d
            weights = weight_tensor.flatten().tolist()
            weights_counter = Counter(weights)

            # build huffman tree
            heap = [[weight, [symbol, ""]] for symbol, weight in weights_counter.items()]
            heapify(heap)
            while len(heap) > 1:
                lo = heappop(heap)
                hi = heappop(heap)
                for pair in lo[1:]:
                    pair[1] = '0' + pair[1]
                for pair in hi[1:]:
                    pair[1] = '1' + pair[1]
                heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
            huffman_tree = heappop(heap)[1:]

            huffman_dict = {symbol: code for symbol, code in huffman_tree}
            encoded_weights = [huffman_dict[weight] for weight in weights]

            for weight in weights:
                huffman_size_bits += len(huffman_dict[weight])

            layer_huffman_info[layer_name]['encoded_weights'] = encoded_weights
            layer_huffman_info[layer_name]['huffman_tree'] = huffman_tree
            layer_huffman_info[layer_name]['shape'] = list(weight_tensor.shape)

    huffman_size_bytes = huffman_size_bits/8
    print("huffman used bytes: ", huffman_size_bytes)

    # dict: layer_huffman_info
    with open('chop/huffman_info.pkl', 'wb') as f:
        pickle.dump(layer_huffman_info, f)

    keys_to_replace = []
    for node in new_graph.nodes:
        if isinstance(get_node_actual_target(node), torch.nn.modules.Conv2d): 
            if 'mase' in node.meta:
                key_to_replace = '.'.join(('.'.join(node.name.rsplit('_', 1))).split('_', 1)) + ".weight"
                keys_to_replace.append(key_to_replace)
    
    #  keys replaced by huffman coding has value of tensor shape instead of tensor itself, so to reduce storage 
    huffman_state_dict = {key: torch.tensor(list(value.shape)) if key in keys_to_replace else value for key, value in new_graph.model.state_dict().items()}

    torch.save(huffman_state_dict, "chop/huffman_model.ckpt")

    return layer_huffman_info

    