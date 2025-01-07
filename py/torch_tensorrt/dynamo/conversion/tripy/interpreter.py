from typing import Dict, List, Sequence

import torch

from .conversion_registry import REGISTRY
from .ops import *  # Needed to register ops

from .executable_module import TorchTripyModule
from torch_tensorrt._Input import Input


# TODO (pranavm): Need to decide if we want to take the current string-based
# approach or construct the trace directly. The upside of the former is that
# we can more easily edit the resulting Tripy program and doesn't require us to
# stabilize the Trace API.
class TripyInterpreter(torch.fx.Interpreter):
    def __init__(self, module: torch.fx.GraphModule, input_specs: Sequence[Input]):
        super().__init__(module)

        self._cur_node_name: str = None
        self._input_specs = input_specs

        self._inputs: List[str] = []
        # Maps tensor names to strings containing code to generate them.
        self._tensor_map: Dict[str, str] = {}
        self._outputs: List[str] = []

    def run_node(self, n: torch.fx.Node):
        # TODO (pranavm): Probably want to use `get_node_name` here
        self._cur_node_name = str(n)
        return super().run_node(n)

    def placeholder(self, target, args, kwargs):
        self._inputs.append(self._cur_node_name)
        return self._cur_node_name

    def get_attr(self, target, args, kwargs):
        raise NotImplementedError()

    def call_function(self, target, args, kwargs):
        # TODO (pranavm): Handle case where op is missing
        self._tensor_map[self._cur_node_name] = REGISTRY[target](args, kwargs)
        return self._cur_node_name

    def call_method(self, target, args, kwargs):
        raise NotImplementedError()

    def call_module(self, target, args, kwargs):
        raise NotImplementedError()

    def output(self, target, args, kwargs):
        assert len(args) == 1
        assert not kwargs
        self._outputs.extend(args[0])
        return args[0]

    def run(self):
        super().run()

        # This import is required to make 'tp' local and accessible under `exec`
        import nvtripy as tp

        tripy_func_str = f"def tripy_func({', '.join(self._inputs)}):\n"

        for lhs, rhs in self._tensor_map.items():
            tripy_func_str += f"    {lhs} = {rhs}\n"

        tripy_func_str += f"    return {', '.join(self._outputs)}\n"

        # TODO (pranavm): Remove debug
        print(tripy_func_str)

        exec(tripy_func_str, locals(), globals())

        # TODO (pranavm): Handle dynamic shaped inputs here.
        compiled_func = tp.compile(
            tripy_func,  # Defined by the string we exec
            args=[
                tp.InputInfo(inp.shape, inp.dtype.to(tp.dtype))
                for inp in self._input_specs
            ],
        )
        return TorchTripyModule(compiled_func)
