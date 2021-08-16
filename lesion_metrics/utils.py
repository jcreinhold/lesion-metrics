# -*- coding: utf-8 -*-
"""
lesion_metrics.utils

utilities for computing lesion metrics

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 16 Aug 2021
"""

__all__ = [
    "bbox",
    "to_numpy",
]

from itertools import combinations
from typing import List

from lesion_metrics.types import Label


def bbox(label: Label) -> List[slice]:
    ndim = label.ndim
    assert isinstance(ndim, int)
    dims = reversed(range(ndim))
    indices: List[slice] = []
    for dim in combinations(dims, ndim - 1):
        nonzero = label.any(dim).nonzero()
        is_numpy = isinstance(nonzero, tuple)
        nonzero = nonzero[0] if is_numpy else nonzero.squeeze()
        indices.append(slice(nonzero[0], nonzero[-1] + 1))
    return indices


def to_numpy(label: Label) -> Label:
    if hasattr(label, "numpy"):
        label = label.numpy()  # type: ignore[attr-defined]
    return label
