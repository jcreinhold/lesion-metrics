"""Calculate total lesion burden in a segmentation mask

calculation of lesion burden/volume given some
type of medical image in a designated volume unit

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: 16 Aug 2021
"""

from __future__ import annotations

import enum
import functools
import operator

import medio.image as mioi

import lesion_metrics.typing as lmt


class UnitOfVolume(enum.Enum):
    microlitre = "micro"
    microliter = "micro"
    milliliter = "milli"
    millilitre = "milli"
    liter = "litre"
    litre = "litre"


class SegmentationVolume:
    def __init__(
        self,
        label: mioi.Image,
        unit: UnitOfVolume = UnitOfVolume.microliter,
    ):
        self.label = label
        self.unit = unit

    @classmethod
    def from_filename(cls, path: lmt.PathLike) -> SegmentationVolume:
        label = mioi.Image.from_path(path)
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
        # per_voxel_volume will be in microliters
        per_voxel_volume = functools.reduce(operator.mul, self.label.spacing, 1.0)
        n_positive_voxels = (self.label > 0.0).sum()
        volume_in_microliters: float = n_positive_voxels * per_voxel_volume
        return volume_in_microliters
