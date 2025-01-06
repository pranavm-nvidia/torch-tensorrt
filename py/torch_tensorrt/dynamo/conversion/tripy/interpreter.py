from typing import List

import torch


class TripyInterpreter(torch.fx.Interpreter):
    def __init__(self, module: torch.fx.GraphModule):
        super().__init__(module)

        self._cur_node_name: str = None
        self.inputs: List[str] = []

    def run_node(self, n: torch.fx.Node):
        # TODO (pranavm): Probably want to use `get_node_name` here
        self._cur_node_name = str(n)
        return super().run_node(n)

    def placeholder(self, target, args, kwargs):
        self.inputs.append(self._cur_node_name)
        # TODO (pranavm): Figure out how to set optimization profiles here.
        return self._cur_node_name

    def get_attr(self, target, args, kwargs):
        raise NotImplementedError()

    def call_function(self, target, args, kwargs):
        raise NotImplementedError()

    def call_method(self, target, args, kwargs):
        raise NotImplementedError()

    def call_module(self, target, args, kwargs):
        raise NotImplementedError()

    def output(self, target, args, kwargs):
        raise NotImplementedError()
