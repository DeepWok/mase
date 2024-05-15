# MaseGraph Passes

## MaseGraph

MaseGraph is defined at `./graph/mase_graph.py`

There are four main components in `MaseGraph`

```python
graph.model # this is the symbolic traced model
graph.fx_graph # this is the fx_graph
graph.modules # This is all modules; TODO: Maybe simplify this? Now all module keys are included recursively.
graph.tracer # TODO: implement this, this is to support dynamic re-tracing
```

`./common.py` also contains the definitions of `MASE_TYPES` and `MASE_OPS`, these are later used in our Pass system.

---

## Pass format

Passes take a format of `Analysis`, `Transform` or a composition between the two.

`./analysis/*.py` contains several existing analysis passes:

1.  `init_metadata` (`./analysis/init_metadata.py`): this pass adds `MaseMetadata` to each node on the `fx_graph`; this is the first pass that is called by default.
2. `add_common_metadata` (`./analysis/add_common_metadata.py`): this pass first tags each node with `mase_op` and `mase_type`, it then adds metadata to all `mase_ops`s, this pass is also called by default.
	* [ ] add more `mase_op` common data.

`./transforms/*.py` contains several existing transform passes:

1. `quantize_transform_pass`: this aims to add quantisation to nodes in the graph.
	* [x] deal with module_related_func
	* [x] quantise funcs
	* [x] add quantisation meta data
	* [x] add another graph iterator to report quantisation
2. `retrace_transform_pass`: this aims to ask tracer to reproduce a graph with custom leaf nodes.


## Testing commands

```bash
# This test uses a funky model that has several built-in funcs
# machop/chop/models/toy_custom_fn.py
./ch transform --config configs/test/transform.toml --model toy-fn
```
