# How to write documentations in MASE

For developers working on large software projects, documentation is essential as it provides a detailed guide to the software's architecture, design, and coding standards, ensuring consistent development practices across the team. It helps in understanding complex systems, facilitates smoother collaboration among team members, and serves as a reference for troubleshooting and making informed decisions about code modifications. Effective documentation also aids in onboarding new developers, enabling them to quickly become productive members of the team, and ensures that the knowledge about the software's intricacies is not lost over time, crucial for long-term maintenance and scalability of the project. 

Here we introduce the documentation rules for MASE.

## The sphinx flow

Documentations are automatically generated from the [docstring](https://peps.python.org/pep-0257/) from each function using [sphinx](https://www.sphinx-doc.org/en/master/).

You can easily generate a local copy of the documentation website for testing whether your doc has been written correctly.

```bash
machop/sphinx_docs
make html
```
This would then build a doc directory under `sphinx_docs/build`, and you can open `sphinx_docs/build/html` and open the `index.html` page to have a local version of the doc.

## Automatic documentation generation through docstrings

### Documentation format

For those who are using VSCode, VSCode already has a plugin named [autoDocstring - Python Docstring Generator](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) that automatically generates a doc template.
We follow the same format for all the passes in MASE. For others, you can just copy the template from existing passes and update the content.
Here is an example:

```python
def some_pass(mg: MaseGraph, pass_args = {...}):
    """_summary_

    :param graph: a MaseGraph
    :type graph: MaseGraph
    :param pass_args: _input_parameters_
    :type pass_args: _type_, optional, 
    :return: _return_parameters_
    :rtype: tuple(MaseGraph, Dict)

    _detailed_descriptions_
    """
    logger.debug(graph.fx_graph)
    graph = graph_iterator_for_mase_ops(graph)
    graph = graph_iterator_for_metadata(graph, **pass_args)
    return graph, {}
```

Here we can fill the details into the template:
* `_summary_`: a single sentence to describe what this pass does, _e.g._ `Add common metadata`.
* `_input_parameters_`: list a set of parameters that can be taken by the pass and give a description for each of them, _e.g._ `"add_value" controls whether tensor values would be added to the meta data, defaults to True; ...`
* `_return_parameters_`: list a set of parameters extracted by the pass as outputs and give a description for each of them. This is similar to `_input_parameters_` but they are output (often seen in analysis passes).
* `_detailed_descriptions_`: give a detailed description of the pass, including motivation, use cases, dependences on other passes, data structures introduced by the pass, examples of inputs and outputs.

A complete example of a pass doc can be found [here](https://github.com/DeepWok/mase/blob/main/machop/chop/passes/graph/analysis/add_metadata/add_common_metadata.py#L226-L378).

### Add the pass to the sphinx directory

The MASE sphinx system only keeps track of the MASE passes, instead of all the Python functions in the codebase.
In order to add your pass to the sphinx system so that it can be displayed on the website, you need to add your pass to the pass list.

The pass list is under [this folder](https://github.com/DeepWok/mase/tree/main/machop/sphinx_docs/source/modules/api). Find the right category and append your pass to the list. 
For example, here we are adding `add_common_metadata` pass, which is a analysis pass belonging to `add_metadata` group, so we find the rst file at `analysis/add_metadata.rst`, and append our pass here:

```rst
chop.passes.graph.analysis.add\_metadata
========================================

add\_common\_metadata\_analysis\_pass
-------------------------------------

.. autofunction:: chop.passes.graph.analysis.add_metadata.add_common_metadata.add_common_metadata_analysis_pass

```

The first line is the name of the pass and the `autofunction` points to the Python function in the codebase.

You also need to add the same information to the top-level pass summary at `passes.rst`, following the same format in the file:

```rst
MaseGraph Analysis Passes
-------------------------

.. list-table:: A summary of all MaseGraph analysis passes 
  :widths: 25 75
  :header-rows: 1

  * - Pass Name
    - Summary
  * - :py:meth:`~chop.passes.graph.analysis.init_metadata.init_metadata_analysis_pass`
    - Initialize each node with the MaseMetadata, this nees to run first before adding any metadata
  * - :py:meth:`~chop.passes.graph.analysis.add_metadata.add_common_metadata.add_common_metadata_analysis_pass`
    - Add metadata used for both software and hardware, this nees to run first before calling to add_software_metadata or add_hardware_metadata
  * - ...
```

Now try to rebuild the doc by following steps in *The sphinx flow*. You will see your new pass is online!

If you have any issues or questions about adding docs, please raise an issue and we will look into that. Thanks!
