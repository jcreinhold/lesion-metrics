"""Evaluate all metrics on a pair of lesion segmentations
Author: Jacob Reinhold
"""

from __future__ import annotations

import builtins
import dataclasses

import medio.image as mioi

import lesion_metrics.metrics as lmm
import lesion_metrics.typing as lmt
import lesion_metrics.volume as lmv


@dataclasses.dataclass
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
        cls,
        pred_filename: lmt.PathLike,
        truth_filename: lmt.PathLike,
        *,
        iou_threshold: builtins.float = 0.0
    ) -> Metrics:
        assert 0.0 <= iou_threshold < 1.0
        pred = mioi.Image.from_path(pred_filename)
        truth = mioi.Image.from_path(truth_filename)
        _avd = lmm.avd(pred, truth)
        dc = lmm.dice(pred, truth)
        jc = lmm.jaccard(pred, truth)
        __lfdr = lmm.lfdr(
            pred, truth, iou_threshold=iou_threshold, return_pred_count=True
        )
        assert isinstance(__lfdr, tuple)
        _lfdr, np = __lfdr
        __ltpr = lmm.ltpr(
            pred, truth, iou_threshold=iou_threshold, return_truth_count=True
        )
        assert isinstance(__ltpr, tuple)
        _ltpr, nt = __lfdr
        _ppv = lmm.ppv(pred, truth)
        _tpr = lmm.tpr(pred, truth)
        isbi15 = lmm.isbi15_score_from_metrics(dc, _ppv, _lfdr, _ltpr)
        vol_t = lmv.SegmentationVolume(truth).volume()
        vol_p = lmv.SegmentationVolume(pred).volume()
        return cls(_avd, dc, isbi15, jc, _lfdr, _ltpr, _ppv, _tpr, vol_t, vol_p, nt, np)
