# -*- coding: utf-8 -*-
"""
lesion_metrics.metrics

holds metrics to evaluate lesion segmentations

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)
Created on: May 14, 2021
"""

__all__ = [
    "dice",
    "jaccard",
    "ppv",
    "tpr",
    "lfpr",
    "ltpr",
    "avd",
    "assd",
    "corr",
    "isbi15_score",
]

from typing import List

from scipy.stats import pearsonr
from skimage.measure import label

from lesion_metrics.types import Label, NaN


def dice(pred: Label, truth: Label) -> float:
    """ dice coefficient between predicted and true binary masks """
    p, t = (pred > 0.0), (truth > 0.0)
    intersection = (p & t).sum()
    cardinality = p.sum() + t.sum()
    if cardinality == 0.0:
        return NaN
    return 2 * intersection / cardinality


def jaccard(pred: Label, truth: Label) -> float:
    """ jaccard index (IoU) between predicted and true binary masks """
    p, t = (pred > 0.0), (truth > 0.0)
    intersection = (p & t).sum()
    union = (p | t).sum()
    if union == 0.0:
        return NaN
    return intersection / union


def ppv(pred: Label, truth: Label) -> float:
    """ positive predictive value (precision) btwn predicted and true binary masks """
    p, t = (pred > 0.0), (truth > 0.0)
    intersection = (p & t).sum()
    denom = p.sum()
    if denom == 0.0:
        return NaN
    return intersection / denom


def tpr(pred: Label, truth: Label) -> float:
    """ true positive rate (sensitivity) between predicted and true binary masks """
    p, t = (pred > 0.0), (truth > 0.0)
    intersection = (p & t).sum()
    denom = t.sum()
    if denom == 0.0:
        return NaN
    return intersection / denom


def lfpr(pred: Label, truth: Label) -> float:
    """ lesion false positive rate between predicted and true binary masks """
    p, t = (pred > 0.0), (truth > 0.0)
    cc, n = label(p, return_num=True)
    if n == 0:
        return NaN
    count = 0
    for i in range(1, n + 1):
        if ((cc == i) & t).sum() == 0:
            count += 1
    return count / n


def ltpr(pred: Label, truth: Label) -> float:
    """ lesion true positive rate between predicted and true binary masks """
    p, t = (pred > 0.0), (truth > 0.0)
    cc, n = label(t, return_num=True)
    if n == 0:
        return NaN
    count = 0
    for i in range(1, n + 1):
        if ((cc == i) & p).sum() > 0.0:
            count += 1
    return count / n


def avd(pred: Label, truth: Label) -> float:
    """ absolute volume difference between predicted and true binary masks """
    p, t = (pred > 0.0), (truth > 0.0)
    numer = abs(p.sum() - t.sum())
    denom = t.sum()
    if denom == 0.0:
        return NaN
    return numer / denom


def assd(pred: Label, truth: Label) -> float:
    """ average symmetric surface difference between predicted and true binary masks """
    raise NotImplementedError


def corr(pred_vols: List[Label], truth_vols: List[Label]) -> float:
    """ pearson correlation coefficient btwn list of predicted and true binary vols """
    return pearsonr(pred_vols, truth_vols)[0]


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
        + (1 - lfpr(pred, truth)) / 4
        + ltpr(pred, truth) / 4
    )
    if reweighted:
        score *= 4 / 3
    return score
