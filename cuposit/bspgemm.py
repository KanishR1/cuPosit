from torch import Tensor

__all__ = ['bspgemm']

try:
    from . import _CUDA
except ImportError:
    raise ImportError(
        "cuposit C++ extension not found. "
        "Please install cuposit properly: pip install -e ."
    )


def bspgemm(
    posit_config: dict[str, int] | int,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tensor:
    def detach(x: Tensor) -> Tensor:
        return x.detach().contiguous().clone()
    
    posit_config_dict: dict[str, int] = {}
    if type(posit_config) is int: 
        posit_config_dict = {
            'n': posit_config,
            'es': 2,
            'rs': posit_config - 1
        }
    else:
        posit_config_dict = posit_config # pyright: ignore[reportAssignmentType]
    
    if 'rs' not in posit_config_dict or posit_config_dict['rs'] is None:
        posit_config_dict['rs'] = posit_config_dict['n'] - 1
    if 'es' not in posit_config_dict or posit_config_dict['es'] is None:
        posit_config_dict['es'] = 2

    if posit_config_dict['n'] >= 4:
        _A, _B, _C = detach(A), detach(B), detach(C)
         
        result = _CUDA.bspgemm(
            _A, _B, _C,
            alpha, beta,
            posit_config_dict['n'], posit_config_dict['es'], posit_config_dict['rs']
        )

        del _A
        del _B

        return result

    raise ValueError(f"Invalid Posit configuration: {posit_config}. See Usage section of readme.")