#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lesion_metrics.lesion_metrics.types

types for lesion_metrics
(support np.ndarray or pytorch tensors
or anything that implements a `sum` method)

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: May 14, 2021
"""

__all__ = [
    "Label",
    "NaN",
]

NaN = float("nan")


class Label:
    """ support anything that implements the methods here """

    def __gt__(self):
        ...

    def sum(self) -> float:
        ...
