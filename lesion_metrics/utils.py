"""Utilities for lesion metrics
Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 16 Aug 2021
"""

__all__ = [
    "bbox",
    "to_numpy",
]

import builtins
import itertools
import typing

import lesion_metrics.typing as lmt


def bbox(label: lmt.Label) -> typing.List[builtins.slice]:
    ndim = label.ndim
    assert isinstance(ndim, int)
    dims = reversed(range(ndim))
    indices: typing.List[builtins.slice] = []
    for dim in itertools.combinations(dims, ndim - 1):
        nonzero = label.any(dim).nonzero()
        is_numpy = isinstance(nonzero, tuple)
        nonzero = nonzero[0] if is_numpy else nonzero.squeeze()
        indices.append(slice(nonzero[0], nonzero[-1] + 1))
    return indices


def to_numpy(label: lmt.Label) -> lmt.Label:
    if hasattr(label, "numpy"):
        label = label.numpy()  # type: ignore[attr-defined]
    return label
