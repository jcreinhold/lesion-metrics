"""Project-specific types

types for lesion_metrics
(support np.ndarray or pytorch tensors
or anything that implements a `sum` method)

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 14, 2021
"""

from __future__ import annotations

import builtins
import os
import typing

__all__ = [
    "Label",
    "NaN",
    "PathLike",
]

NaN = float("nan")
PathLike = typing.Union[os.PathLike, builtins.str]


class Label(typing.Protocol):
    """support anything that implements the methods here"""

    def __gt__(self, other: typing.Any) -> typing.Any:
        ...

    def __and__(self, other: Label) -> Label:
        ...

    def __or__(self, other: Label) -> Label:
        ...

    def __getitem__(
        self, item: typing.Union[typing.Tuple[builtins.slice, ...], builtins.int]
    ) -> typing.Any:
        ...

    def sum(self) -> builtins.float:
        ...

    @property
    def ndim(self) -> builtins.int:
        ...

    def any(
        self,
        axis: typing.Optional[
            typing.Union[builtins.int, typing.Tuple[builtins.int, ...]]
        ] = None,
    ) -> typing.Any:
        ...

    def nonzero(self) -> typing.Any:
        ...

    def squeeze(self) -> typing.Any:
        ...
