import torch
from ..conversion_registry import REGISTRY


@REGISTRY.register(torch.ops.aten.abs.default)
def aten_abs(args, kwargs):
    assert len(args) == 1
    assert not kwargs

    return f"tp.abs({args[0]})"
