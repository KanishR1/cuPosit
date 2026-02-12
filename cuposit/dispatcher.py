import warnings
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from cuposit import ops as cuposit_ops


class MatMulDispatcher(TorchDispatchMode):
    def __init__(self, posit_config: dict[str, int], enabled: bool=True):
        self.posit_config: dict[str, int] = posit_config
        self.enabled: bool = enabled

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if not self.enabled:
            return func(*args, **kwargs)
                
        if func in (
            torch.ops.aten.mm.default,
        ):
            return cuposit_ops.mm(self.posit_config, *args, **kwargs)

        if func in (
            torch.ops.aten.addmm.default,
        ):
            return cuposit_ops.addmm(self.posit_config, *args, **kwargs)

        if func in (
            torch.ops.aten.convolution.default,
        ):
            return cuposit_ops.convolution(self.posit_config, *args, **kwargs)

        if func in (
            torch.ops.aten.matmul.default,
            torch.ops.aten.bmm.default,
        ):
            warnings.warn(f"Cuposit Dispatcher noticed an uncaught matmul op: {func}")

        return func(*args, **kwargs)