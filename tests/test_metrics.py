#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `lesion_metrics` package."""

from pathlib import Path

import nibabel as nib
import pytest

from lesion_metrics.metrics import (
    dice,
    jaccard,
    ppv,
    tpr,
    lfdr,
    ltpr,
    avd,
    corr,
    isbi15_score,
)
from lesion_metrics.types import Label


@pytest.fixture
def file() -> Path:
    return Path(__file__).resolve()


@pytest.fixture
def cwd(file) -> Path:
    return file.parent


@pytest.fixture
def pred(cwd: Path) -> Label:
    fn = cwd / "test_data" / "pred" / "pred.nii.gz"
    return nib.load(fn).get_fdata()


@pytest.fixture
def truth(cwd: Path) -> Label:
    fn = cwd / "test_data" / "truth" / "truth.nii.gz"
    return nib.load(fn).get_fdata()


def test_dice(pred: Label, truth: Label):
    dice_coef = dice(pred, truth)
    correct = 2 * (3 / ((8 + 1 + 1) + (2 + 1 + 1)))
    assert dice_coef == correct


def test_jaccard(pred: Label, truth: Label):
    jaccard_idx = jaccard(pred, truth)
    correct = 3 / ((8 + 1 + 1) + 1)
    assert jaccard_idx == correct


def test_ppv(pred: Label, truth: Label):
    ppv_score = ppv(pred, truth)
    correct = 3 / (2 + 1 + 1)
    assert ppv_score == correct


def test_tpr(pred: Label, truth: Label):
    tpr_score = tpr(pred, truth)
    correct = 3 / (8 + 1 + 1)
    assert tpr_score == correct


def test_lfdr(pred: Label, truth: Label):
    lfpr_score = lfdr(pred, truth)
    correct = 1 / 3
    assert lfpr_score == correct


def test_ltpr(pred: Label, truth: Label):
    ltpr_score = ltpr(pred, truth)
    correct = 2 / 3
    assert ltpr_score == correct


def test_avd(pred: Label, truth: Label):
    avd_score = avd(pred, truth)
    correct = 0.6
    assert avd_score == correct


def test_corr(pred: Label, truth: Label):
    ps = pred.sum()
    ts = truth.sum()
    eps = 0.1
    pred_vols = [ps, ps + eps, ps - eps]
    truth_vols = [ts, ts + eps, ts - eps]
    corr_score = corr(pred_vols, truth_vols)
    correct = 1.0
    assert pytest.approx(corr_score, 1e-3) == correct


def test_isbi15_score(pred: Label, truth: Label):
    isbi15 = isbi15_score(pred, truth)
    correct = 0.6408730158730158
    assert isbi15 == correct


@pytest.mark.skip("Not implemented.")
def test_assd(pred: Label, truth: Label):
    pass
