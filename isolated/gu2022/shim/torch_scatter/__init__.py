"""Módulo de compatibilidad mínimo de `torch_scatter` para el código de Gu et al. (2022).

Solo implementa lo que usa `gcnn/model.py`: `scatter_mean` y `scatter_min` (y `scatter`, que
aparece en un bloque comentado), con `torch.Tensor.scatter_reduce` de PyTorch. Semántica de
`torch_scatter` 2.x: las posiciones sin elementos valen 0; `scatter_min` devuelve
(mínimos, índice del mínimo), con índice = src.size(dim) en las posiciones vacías.

Se usa SOLO en el entorno aislado de F3.5 para medir tiempos; para la reproducción se
compilará el `torch_scatter` original (o se declarará este módulo como cambio mínimo).
"""

from __future__ import annotations

import torch


def _expand_index(index: torch.Tensor, src: torch.Tensor, dim: int) -> torch.Tensor:
    """Expande `index` (1-D, a lo largo de `dim`) a la forma de `src`, como torch_scatter."""
    dim = dim % src.dim()
    shape = [1] * src.dim()
    shape[dim] = -1
    return index.view(shape).expand_as(src)


def _out(src: torch.Tensor, dim: int, dim_size: int | None, index: torch.Tensor) -> torch.Tensor:
    """Tensor de salida con ceros, del tamaño pedido en `dim`."""
    dim = dim % src.dim()
    size = list(src.shape)
    size[dim] = int(dim_size) if dim_size is not None else int(index.max()) + 1
    return torch.zeros(size, dtype=src.dtype, device=src.device)


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    """Reducción por grupos (`sum`, `mean`, `min`, `max`); vacías = 0."""
    mode = {"sum": "sum", "add": "sum", "mean": "mean", "min": "amin", "max": "amax"}[reduce]
    idx = _expand_index(index, src, dim)
    base = _out(src, dim, dim_size, index) if out is None else out
    return base.scatter_reduce(dim % src.dim(), idx, src, reduce=mode, include_self=False)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    """Media por grupos; los grupos vacíos valen 0."""
    return scatter(src, index, dim, out, dim_size, reduce="mean")


def scatter_min(src, index, dim=-1, out=None, dim_size=None):
    """(mínimo por grupo, posición del mínimo); grupos vacíos: (0, src.size(dim))."""
    d = dim % src.dim()
    values = scatter(src, index, d, out, dim_size, reduce="min")
    # Posición del mínimo: la menor posición cuyo valor coincide con el mínimo de su grupo.
    idx = _expand_index(index, src, d)
    pos_shape = [1] * src.dim()
    pos_shape[d] = -1
    pos = torch.arange(src.size(d), device=src.device).view(pos_shape).expand_as(src)
    is_min = src == values.gather(d, idx)
    big = torch.full_like(pos, src.size(d))
    cand = torch.where(is_min, pos, big)
    arg = torch.full(values.shape, src.size(d), dtype=torch.long, device=src.device)
    arg = arg.scatter_reduce(d, idx, cand, reduce="amin", include_self=True)
    return values, arg
