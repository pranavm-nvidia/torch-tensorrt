from torch.fx.node import Target
from typing import Dict, Callable, List


OpConverter = Callable[[List[str], Dict[str, str]], str]


class ConversionRegistry(dict):
    def register(self, target: Target):
        def impl(func: OpConverter):
            self[target] = func
            return func

        return impl


REGISTRY = ConversionRegistry()
