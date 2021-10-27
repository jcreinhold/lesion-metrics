#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `lesion_metrics` package."""

from pathlib import Path

import nibabel as nib
import pytest
import torch

from lesion_metrics.metrics import (
    avd,
    corr,
    dice,
    isbi15_score,
    jaccard,
    lfdr,
    ltpr,
    ppv,
    tpr,
)
from lesion_metrics.types import Label
from lesion_metrics.volume import SegmentationVolume


@pytest.fixture
def cwd() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "tests":
        return cwd
    cwd = (cwd / "tests").resolve(strict=True)
    return cwd


@pytest.fixture(params=("numpy", "torch"))
def backend(request) -> str:  # type: ignore[no-untyped-def]
    _backend: str = request.param
    return _backend


@pytest.fixture
def pred_filename(cwd: Path) -> Path:
    return cwd / "test_data" / "pred" / "pred.nii.gz"


@pytest.fixture
def pred(backend: str, pred_filename: Path) -> Label:
    pred_data: Label = nib.load(pred_filename).get_fdata()
    if backend == "torch":
        pred_data = torch.from_numpy(pred_data)  # type: ignore[assignment]
    return pred_data


@pytest.fixture
def truth_filename(cwd: Path) -> Path:
    return cwd / "test_data" / "truth" / "truth.nii.gz"


@pytest.fixture
def truth(backend: str, truth_filename: Path) -> Label:
    truth_data: Label = nib.load(truth_filename).get_fdata()
    if backend == "torch":
        truth_data = torch.from_numpy(truth_data)  # type: ignore[assignment]
    return truth_data


def test_dice(pred: Label, truth: Label) -> None:
    dice_coef = dice(pred, truth)
    correct = 2 * (3 / ((8 + 1 + 1) + (2 + 1 + 1)))
    assert dice_coef == correct


def test_jaccard(pred: Label, truth: Label) -> None:
    jaccard_idx = jaccard(pred, truth)
    correct = 3 / ((8 + 1 + 1) + 1)
    assert jaccard_idx == correct


def test_ppv(pred: Label, truth: Label) -> None:
    ppv_score = ppv(pred, truth)
    correct = 3 / (2 + 1 + 1)
    assert ppv_score == correct


def test_tpr(pred: Label, truth: Label) -> None:
    tpr_score = tpr(pred, truth)
    correct = 3 / (8 + 1 + 1)
    assert tpr_score == correct


def test_lfdr(pred: Label, truth: Label) -> None:
    lfpr_score = lfdr(pred, truth)
    correct = 1 / 3
    assert lfpr_score == correct


def test_ltpr(pred: Label, truth: Label) -> None:
    ltpr_score = ltpr(pred, truth)
    correct = 2 / 3
    assert ltpr_score == correct


def test_avd(pred: Label, truth: Label) -> None:
    avd_score = avd(pred, truth)
    correct = 0.6
    assert avd_score == correct


def test_corr(pred: Label, truth: Label) -> None:
    ps = pred.sum()
    ts = truth.sum()
    eps = 0.1
    pred_vols = [ps, ps + eps, ps - eps]
    truth_vols = [ts, ts + eps, ts - eps]
    corr_score = corr(pred_vols, truth_vols)
    correct = 1.0
    assert pytest.approx(corr_score, 1e-3) == correct


def test_isbi15_score(pred: Label, truth: Label) -> None:
    isbi15 = isbi15_score(pred, truth)
    correct = 0.6408730158730158
    assert isbi15 == pytest.approx(correct, 1e-3)


def test_segmentation_volume(pred_filename: Path) -> None:
    sv = SegmentationVolume.from_filename(pred_filename)
    vol = sv.volume()
    assert vol == 4.0


@pytest.mark.skip("Not implemented.")
def test_assd(pred: Label, truth: Label) -> None:
    pass
