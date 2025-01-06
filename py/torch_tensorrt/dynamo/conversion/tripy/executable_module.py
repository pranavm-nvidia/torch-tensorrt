import torch
import nvtripy as tp

from typing import Sequence


class TorchTripyModule(torch.nn.Module):
    def __init__(self, compiled_func):
        super().__init__()
        self.compiled_func = compiled_func

    def forward(self, *args, **kwargs):
        def wrap_torch_tensor(arg):
            if isinstance(arg, torch.Tensor):
                return tp.Tensor(arg)
            return arg

        def unwrap_tripy_tensor(arg):
            if isinstance(arg, tp.Tensor):
                return torch.from_dlpack(arg)
            return arg

        new_args = [wrap_torch_tensor(arg) for arg in args]
        new_kwargs = {key: wrap_torch_tensor(value) for key, value in kwargs.items()}

        outs = self.compiled_func(*new_args, *new_kwargs)
        if isinstance(outs, Sequence):
            return [unwrap_tripy_tensor(out) for out in outs]
        return [unwrap_tripy_tensor(outs)]
