# -*- coding: utf-8 -*-
"""
lesion_metrics.helper

holds helper class to evaluate all metrics on a pair of
lesion segmentations

Author: Jacob Reinhold
Created on: 09 Dec 2021
"""

from dataclasses import dataclass
from typing import Type, TypeVar

import torchio as tio

from lesion_metrics.metrics import (
    avd,
    dice,
    isbi15_score_from_metrics,
    jaccard,
    lfdr,
    ltpr,
    ppv,
    tpr,
)
from lesion_metrics.types import PathLike
from lesion_metrics.volume import SegmentationVolume

M = TypeVar("M", bound="Metrics")


@dataclass
class Metrics:
    avd: float
    dice: float
    isbi15_score: float
    jaccard: float
    lfdr: float
    ltpr: float
    ppv: float
    tpr: float
    truth_volume: float
    pred_volume: float
    truth_count: int
    pred_count: int

    @classmethod
    def from_filenames(
        cls: Type[M],
        pred_filename: PathLike,
        truth_filename: PathLike,
        *,
        iou_threshold: float = 0.0
    ) -> M:
        assert 0.0 <= iou_threshold < 1.0
        pred = tio.LabelMap(pred_filename)
        truth = tio.LabelMap(truth_filename)
        _avd = avd(pred, truth)
        dc = dice(pred, truth)
        jc = jaccard(pred, truth)
        __lfdr = lfdr(pred, truth, iou_threshold=iou_threshold, return_pred_count=True)
        assert isinstance(__lfdr, tuple)
        _lfdr, np = __lfdr
        __ltpr = ltpr(pred, truth, iou_threshold=iou_threshold, return_truth_count=True)
        assert isinstance(__ltpr, tuple)
        _ltpr, nt = __lfdr
        _ppv = ppv(pred, truth)
        _tpr = tpr(pred, truth)
        isbi15 = isbi15_score_from_metrics(dc, _ppv, _lfdr, _ltpr)
        vol_t = SegmentationVolume(truth).volume()
        vol_p = SegmentationVolume(pred).volume()
        return cls(_avd, dc, isbi15, jc, _lfdr, _ltpr, _ppv, _tpr, vol_t, vol_p, nt, np)
