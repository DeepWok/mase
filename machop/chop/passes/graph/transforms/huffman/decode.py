def huffman_decode_pass (
        layer_huffman_info
    ):
    # decode

    import torch
    import pickle
    import pdb
    from chop.passes.graph.utils import get_mase_op, get_node_actual_target

    import gc
    gc.collect()

    with open('chop/huffman_info.pkl', 'rb') as f:
        layer_huffman_info = pickle.load(f)

    def decode_huffman(encoded_weights, huffman_tree):
        reverse_huffman_tree = {code: weight for weight, code in huffman_tree}
        decoded_weights = []
        for encoded_weight in encoded_weights:
            decoded_weights.append(reverse_huffman_tree[encoded_weight])
        return decoded_weights

    decoded_tensor = {}

    for layer_name in layer_huffman_info:
        encoded_weights = layer_huffman_info[layer_name]['encoded_weights']
        huffman_tree = layer_huffman_info[layer_name]['huffman_tree']
        shape = layer_huffman_info[layer_name]['shape']
        decoded_weights = decode_huffman(encoded_weights, huffman_tree)

        decoded_layer_tensor = torch.tensor(decoded_weights).reshape(shape[0], shape[1], shape[2], shape[3])
        decoded_tensor[layer_name] = decoded_layer_tensor
    
    return decoded_tensor