from .gelu import (
    Distilled_GELU,
)


replaced_base_module_map = {
    "gelu_sta": Distilled_GELU,
}


replaced_module_map = {
    **replaced_base_module_map,
}