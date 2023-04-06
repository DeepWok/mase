# New Modifier

- `coarse_grained.py`: coarse-grained (module class, function, and method replacement) modify-sw using
  - `module_classes_to_modify.MODULE_CLASS_NAME`, where `MODULE_CLASS_NAME` can be found at `/mase-tools/software/machop/modify/MODULE_CLASS_NAME_TO_MODULE_CLASS`
  - `functions_to_modify.FUNCTION_NAME`, where `FUNCTION_NAME` can be found at `/mase-tools/software/machop/modify/modifier.FUNCTION_NAME_TO_FUNCTIONS`
  - `methods_to_modify.METHOD_NAME`, where `METHOD_NAME` can be found at `/mase-tools/software/machop/modify/modifier.METHOD_NAMES`.
- `fine_grained.py`: fine-grained (layer level replacement) modify-sw using
  - `modules_to_modify.MODULE_NAME`, where `MODULE_NAME` is `fx.Node.target`
- `replace_custom_block.py`: replacement of custom blocks
  - `Modifier` has a class method `Modifier.create_empty_config_template`to generate toml template.
