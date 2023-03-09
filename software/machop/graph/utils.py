def get_module_by_name(model, request_name):
    for name, layer in model.named_modules():
        if name == request_name:
            return layer
    return None


# Verilog format
# Format to a compatible verilog name
def vf(string):
    return string.replace(".", "_").replace(" ", "_")
