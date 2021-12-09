# -*- coding: utf-8 -*-
"""
lesion_metrics.volume

calculation of lesion burden/volume given some
type of medical image in a designated volume unit

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: 16 Aug 2021
"""

from enum import Enum
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Type, TypeVar, Union

import torchio as tio

from lesion_metrics.types import PathLike

SV = TypeVar("SV", bound="SegmentationVolume")


class UnitOfVolume(Enum):
    microlitre = "micro"
    microliter = "micro"
    milliliter = "milli"
    millilitre = "milli"
    liter = "litre"
    litre = "litre"


class SegmentationVolume:
    def __init__(
        self,
        label: tio.Image,
        unit: UnitOfVolume = UnitOfVolume.microliter,
    ):
        self.label = label
        self.unit = unit

    @classmethod
    def from_filename(cls: Type[SV], path: PathLike) -> SV:
        label = tio.LabelMap(path)
        return cls(label)

    def volume(self) -> float:
        vol_in_micro = self.volume_in_microliters()
        if self.unit == UnitOfVolume.microliter:
            return vol_in_micro
        elif self.unit == UnitOfVolume.milliliter:
            return vol_in_micro / 1e3
        elif self.unit == UnitOfVolume.liter:
            return vol_in_micro / 1e6
        else:
            raise ValueError(f"Invalid unit: {self.unit}")

    def volume_in_microlitres(self) -> float:
        return self.volume_in_microliters()

    def volume_in_microliters(self) -> float:
        per_voxel_volume = reduce(mul, self.label.spacing, 1.0)  # in microliters
        n_positive_voxels = (self.label.numpy() > 0.0).sum()
        volume_in_microliters: float = n_positive_voxels * per_voxel_volume
        return volume_in_microliters
