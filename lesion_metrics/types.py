# -*- coding: utf-8 -*-
"""
lesion_metrics.lesion_metrics.types

types for lesion_metrics
(support np.ndarray or pytorch tensors
or anything that implements a `sum` method)

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 14, 2021
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Union

__all__ = [
    "Label",
    "NaN",
    "PathLike",
]

NaN = float("nan")
PathLike = Union[str, Path]


class Label:
    """support anything that implements the methods here"""

    def __gt__(self, other: Union["Label", float]) -> "Label":
        ...

    def __and__(self, other: "Label") -> "Label":
        ...

    def __or__(self, other: "Label") -> "Label":
        ...

    def __getitem__(self, item: Union[Tuple[slice, ...], int]) -> Any:
        ...

    def sum(self) -> float:
        ...

    def ndim(self) -> int:
        ...

    def any(self, axis: Optional[int] = None) -> "Label":
        ...

    def nonzero(self) -> "Label":
        ...

    def squeeze(self) -> "Label":
        ...
