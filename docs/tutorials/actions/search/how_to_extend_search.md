# How to extend search

This tutorial shows how to extend the search space and search strategy in MASE.

## How does MASE search work?

- search_space defines the search space using two dictionaries:
    - `search_space.choices_flattened`: a flatten dictionary of all the choices in the search space
    - `search_space.choice_lengths_flattened`: a flatten dictionary of the lengths of all the choices in the search space
- For every trial,
    1. search_strategy samples from search_space. search_strategy uses `search_space.choice_lengths_flattened` to create `sampled_indexes` for each value in `search_space.choices_flattened`. Then search_strategy uses `search_space.flattened_indexes_to_config(...)` to convert `sampled_indexes` to a `sampled_config`.
    2. search_strategy passes the `sampled_config` to `search_space.rebuild_model(...)` to create a new `model`.
    3. search_strategy passes to `model` to `search_strategy.sw_runner`/`search_strategy.hw_runner` to get the sw/hw metrics.
    4. search_strategy will use the sw/hw metrics to guide next trial.
- In the end, search_strategy will save the researched results.

Note that search_space is search space + model, and search_strategy is search algorithm + dataloader. search_strategy interacts with search_space through `sampled_config`, rebuilt `model`, and `sw_runner`/`hw_runner`.

## How to extend search_space?

- `[Required]` Create a new search space class that inherits from `SearchSpaceBase` at `mase-tools/machop/chop/actions/search/search_space/base.py` and implement corresponding abstract methods.
- `[Required]` Register the new search space to `SEARCH_SPACE_MAP` at `mase-tools/machop/chop/actions/search/search_space/__init__.py`.
- `[Optional]` Add new software (hardware) metrics:
    - subclass `SoftwareRunnerBase` at `mase-tools/machop/chop/actions/search/runners/software/base.py`and implement corresponding abstract methods.
    - Register the new software metrics at `SOFTWARE_RUNNER_MAP` at `mase-tools/machop/chop/actions/search/runners/software/__init__.py`.


# How to extend search_strategy?

🚧 under construction 🚧