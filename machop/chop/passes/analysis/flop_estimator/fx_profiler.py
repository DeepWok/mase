"""
Profiler is expected to be rewritten as interpreter passes.
"""
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from torch.fx import Interpreter
from torch.fx.node import Argument, Target


class Stats:
    data = defaultdict(dict)

    def add_data(self, key, value):
        self.data[key] = value


class GraphProfiler(Interpreter):
    stats = Stats()

    def set_stats(self, stats):
        self.stats = stats

    def set_function_wrapper(self, func_wrap):
        self.call_function_wrap = func_wrap

    def set_module_wrapper(self, module_wrap_func):
        self.call_module_wrap = module_wrap_func

    def call_function(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_function`` node and return the result.
        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        Return
            Any: The value returned by the function invocation
        """
        assert not isinstance(target, str)

        # Execute the function and return the result

        out_data = target(*args, **kwargs)
        # meta = calculate_funcs(target, args, kwargs, out_data)
        # self.stats.add_data(target.__name__, meta)
        meta = self.call_function_wrap(target, args, kwargs, out_data)
        self.stats.add_data(target.__name__, meta)
        return out_data

    def call_module(
        self, target: "Target", args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        """
        Execute a ``call_module`` node and return the result.
        Args:
            target (Target): The call target for this node. See
                `Node <https://pytorch.org/docs/master/fx.html#torch.fx.Node>`__ for
                details on semantics
            args (Tuple): Tuple of positional args for this invocation
            kwargs (Dict): Dict of keyword arguments for this invocation
        Return
            Any: The value returned by the module invocation
        """
        # Retrieve executed args and kwargs values from the environment

        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)

        in_data = args[0]

        out_data = submod(*args, **kwargs)
        # meta = calculate_modules(submod, in_data, out_data)
        meta = self.call_module_wrap(submod, in_data, out_data)
        self.stats.add_data(target, meta)
        return out_data
