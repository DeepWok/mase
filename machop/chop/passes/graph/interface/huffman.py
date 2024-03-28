from collections import Counter
import heapq
import json
import torch
import bitarray, os

from ..utils import get_node_actual_target

def huffman_encode(freqs):
    heap = [[weight, [symbol, ""]] for symbol, weight in freqs.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))

def find_module_of_parameter(model, full_param_name):
    # Split the full parameter name into parts
    parts = full_param_name.split('.')
    submodule_path = parts[:-1]  # Everything except the last part, which is the parameter name
    
    # Start with the base model
    current_module = model
    
    # Traverse the modules according to the path
    for submodule_name in submodule_path:
        # Update the current_module to go deeper
        if hasattr(current_module, submodule_name):
            current_module = getattr(current_module, submodule_name)
        else:
            # Return None if any part of the path doesn't exist
            return None
    
    return current_module

def flatten_parameters(model, mg):
    """Flatten and concatenate all model parameters into a list."""
    named_params = list(model.named_parameters())

    get_named_params = lambda name, params: [param for param in params if name in param[0]]

    param_list = []
    for node in mg.fx_graph.nodes:
        module = get_node_actual_target(node)
        if node.target in mg.modules:
            named_params_node = get_named_params(f"{node.target}.", named_params)      
            for name, _ in module.named_parameters():
                actual_name, actual_param = get_named_params(name, named_params_node)[0]
                actual_module = find_module_of_parameter(model, actual_name)
                # If the model is quantized, apply the quantization
                if hasattr(module, "w_quantizer"):
                    actual_param = module.w_quantizer(actual_param)
                if hasattr(module, "b_quantizer"):
                    actual_param = module.b_quantizer(actual_param)
                # If the module is pruned, apply the mask
                if isinstance(actual_module, torch.nn.utils.parametrize.ParametrizationList) and "weight" in actual_name:
                    actual_param = actual_param * actual_module[0].mask.to(actual_param.dtype)
                param_list.extend(actual_param.flatten().tolist())
    return param_list


def huffman_encode_pass(mg, pass_args):
    model = pass_args['model']

    save_dir = pass_args['save_dir']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, "parameters"))
        os.makedirs(os.path.join(save_dir, "masks"))

    # Flatten all parameters to get a unique set and frequency counter
    params = flatten_parameters(model, mg)
    freqs = Counter(params)

    # Build the huffman tree
    global_huffman_codes = huffman_encode(freqs)

    huffman_codes_path = os.path.join(save_dir, "global_huffman_codes.json")
    with open(huffman_codes_path, 'w') as file:
        json.dump(global_huffman_codes, file)
    
    named_params = list(model.named_parameters())

    get_named_params = lambda name, params: [param for param in params if name in param[0]]
    for node in mg.fx_graph.nodes:
        module = get_node_actual_target(node)
        if node.target in mg.modules:
            named_params_node = get_named_params(f"{node.target}.", named_params)      
            for name, _ in module.named_parameters():
                actual_name, actual_param = get_named_params(name, named_params_node)[0]
                actual_module = find_module_of_parameter(model, actual_name)
                # If the model is quantized, apply the quantization
                if hasattr(module, "w_quantizer"):
                    actual_param = module.w_quantizer(actual_param)
                if hasattr(module, "b_quantizer"):
                    actual_param = module.b_quantizer(actual_param)
                # If the module is pruned, apply the mask
                if isinstance(actual_module, torch.nn.utils.parametrize.ParametrizationList) and "weight" in actual_name:
                    actual_param = actual_param * actual_module[0].mask.to(actual_param.dtype)
                
                actual_param_values = actual_param.flatten().tolist()
                # Encode using global Huffman codes
                encoded_values = ''.join([global_huffman_codes[val] for val in actual_param_values])
                ba = bitarray.bitarray(encoded_values)
                
                # Save encoded parameters
                encoded_file_path = os.path.join(save_dir, "parameters", f"{node.target}.{name}.{len(ba)}.bin")

                with open(encoded_file_path, 'wb') as encoded_file:
                    ba.tofile(encoded_file)

    return mg, {}

def decode_huffman(encoded_data, huffman_codes):
    reverse_huffman_codes = {v: k for k, v in huffman_codes.items()}
    decoded_data = []
    code = ""
    for bit in encoded_data:
        code += str(bit)
        if code in reverse_huffman_codes:
            decoded_data.append(float(reverse_huffman_codes[code]))
            code = ""
    return decoded_data

def load_huffman_encoded_model(mg, pass_args):
    load_dir = pass_args['load_dir']

    model = mg.model

    # Load the huffman codes
    with open(f"{load_dir}/global_huffman_codes.json", 'r') as file:
        huffman_codes = json.load(file)

    encoded_parameter_files = os.listdir(os.path.join(load_dir, "parameters"))

    for name, param in model.named_parameters():
        file_name = None
        # Find the file associated with the parameter
        for file in encoded_parameter_files:
            if file.startswith(name):
                file_name = file
                break
        
        if file_name is None:
            continue
        
        # Load the file
        encoded_file_path = os.path.join(load_dir, "parameters", file_name)
        num_bits = int(file_name.split('.')[-2])
        with open(encoded_file_path, 'rb') as encoded_file:
            encoded_data = bitarray.bitarray()
            # Load the encoded data
            encoded_data.fromfile(encoded_file)
            # Extract the number of bits
            encoded_data = encoded_data[:num_bits]
            # Decode the data
            decoded_data = decode_huffman(encoded_data, huffman_codes)
            decoded_data = torch.tensor(decoded_data)
            decoded_data = decoded_data.view(param.shape)
            param.data = decoded_data

    mg.model = model

    return mg, {}