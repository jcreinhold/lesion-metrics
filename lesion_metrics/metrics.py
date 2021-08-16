# -*- coding: utf-8 -*-
"""
lesion_metrics.metrics

holds metrics to evaluate lesion segmentations

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 14, 2021
"""

__all__ = [
    "assd",
    "avd",
    "corr",
    "dice",
    "iou_per_lesion",
    "isbi15_score",
    "jaccard",
    "lfdr",
    "ltpr",
    "ppv",
    "tpr",
]

from typing import List

from scipy.stats import pearsonr
from skimage.measure import label

from lesion_metrics.types import Label, NaN, Real
from lesion_metrics.utils import bbox, to_numpy


def dice(pred: Label, truth: Label) -> float:
    """dice coefficient between predicted and true binary masks"""
    p, t = (pred > 0.0), (truth > 0.0)
    intersection = (p & t).sum()
    cardinality = p.sum() + t.sum()
    if cardinality == 0.0:
        return NaN
    score: float = 2 * intersection / cardinality
    return score


def jaccard(pred: Label, truth: Label) -> float:
    """jaccard index (IoU) between predicted and true binary masks"""
    p, t = (pred > 0.0), (truth > 0.0)
    intersection = (p & t).sum()
    union = (p | t).sum()
    if union == 0.0:
        return NaN
    score: float = intersection / union
    return score


def ppv(pred: Label, truth: Label) -> float:
    """positive predictive value (precision) btwn predicted and true binary masks"""
    p, t = (pred > 0.0), (truth > 0.0)
    intersection = (p & t).sum()
    denom = p.sum()
    if denom == 0.0:
        return NaN
    score: float = intersection / denom
    return score


def tpr(pred: Label, truth: Label) -> float:
    """true positive rate (sensitivity) between predicted and true binary masks"""
    p, t = (pred > 0.0), (truth > 0.0)
    intersection = (p & t).sum()
    denom = t.sum()
    if denom == 0.0:
        return NaN
    score: float = intersection / denom
    return score


def iou_per_lesion(
    target: Label,
    other: Label,
) -> List[float]:
    """iou of each lesion using target as reference"""
    t, o = (target > 0.0), (other > 0.0)
    cc, n = label(t, return_num=True)
    ious: List[float] = []
    for i in range(1, n + 1):
        lesion_bbox = bbox(cc == i)
        target_lesion = t[lesion_bbox]
        other_lesion = o[lesion_bbox]
        ious.append(jaccard(other_lesion, target_lesion))
    return ious


def lfdr(pred: Label, truth: Label, iou_threshold: float = 0.0) -> float:
    """lesion false discovery rate between predicted and true binary masks"""
    assert iou_threshold >= 0.0
    p, t = (pred > 0.0), (truth > 0.0)
    p, t = to_numpy(p), to_numpy(t)
    ious = iou_per_lesion(p, t)
    if not ious:
        return NaN
    false_positives = [iou <= iou_threshold for iou in ious]
    fp = sum(false_positives)
    fp_plus_tp = len(false_positives)
    score: float = fp / fp_plus_tp
    return score


def ltpr(pred: Label, truth: Label, iou_threshold: float = 0.0) -> float:
    """lesion true positive rate between predicted and true binary masks"""
    assert iou_threshold >= 0.0
    p, t = (pred > 0.0), (truth > 0.0)
    p, t = to_numpy(p), to_numpy(t)
    ious = iou_per_lesion(t, p)
    if not ious:
        return NaN
    true_positives = [iou > iou_threshold for iou in ious]
    tp = sum(true_positives)
    tp_plus_fp = len(true_positives)
    score: float = tp / tp_plus_fp
    return score


def avd(pred: Label, truth: Label) -> float:
    """absolute volume difference between predicted and true binary masks"""
    p, t = (pred > 0.0), (truth > 0.0)
    numer = abs(p.sum() - t.sum())
    denom = t.sum()
    if denom == 0.0:
        return NaN
    score: float = numer / denom
    return score


def assd(pred: Label, truth: Label) -> float:
    """average symmetric surface difference between predicted and true binary masks"""
    raise NotImplementedError


def corr(pred_vols: List[Real], truth_vols: List[Real]) -> float:
    """pearson correlation coefficient btwn list of predicted and true binary vols"""
    coef: float = pearsonr(pred_vols, truth_vols)[0]
    return coef


def isbi15_score(pred: Label, truth: Label, reweighted: bool = True) -> float:
    """
    report the score (minus volume correlation)
    for a given prediction as described in [1]

    reweighted flag puts the score (excluding
    volume correlation which requires a list of
    labels) between 0 and 1

    References:
        [1] Carass, Aaron, et al. "Longitudinal multiple sclerosis
            lesion segmentation: resource and challenge." NeuroImage
            148 (2017): 77-102.
    """
    score = (
        dice(pred, truth) / 8
        + ppv(pred, truth) / 8
        + (1 - lfdr(pred, truth)) / 4
        + ltpr(pred, truth) / 4
    )
    if reweighted:
        score *= 4 / 3
    return score
