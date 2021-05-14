#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `lesion_metrics` package."""

from pathlib import Path

import nibabel as nib
import pytest

from lesion_metrics.metrics import *


@pytest.fixture
def file():
    return Path(__file__).resolve()


@pytest.fixture
def cwd(file):
    return file.parent


@pytest.fixture
def pred(cwd: Path) -> nib.Nifti1Image:
    fn = cwd / 'test_data' / 'pred' / 'pred.nii.gz'
    return nib.load(fn).get_fdata()


@pytest.fixture
def truth(cwd: Path) -> nib.Nifti1Image:
    fn = cwd / 'test_data' / 'truth' / 'truth.nii.gz'
    return nib.load(fn).get_fdata()


def test_dice(pred, truth):
    dice_coef = dice(pred, truth)
    correct = 2 * (3 / ((8 + 1 + 1) + (2 + 1 + 1)))
    assert dice_coef == correct


def test_jaccard(pred, truth):
    jaccard_idx = jaccard(pred, truth)
    correct = (3 / ((8 + 1 + 1) + 1))
    assert jaccard_idx == correct


def test_ppv(pred, truth):
    ppv_score = ppv(pred, truth)
    correct = (3 / (2 + 1 + 1))
    assert ppv_score == correct


def test_tpr(pred, truth):
    tpr_score = tpr(pred, truth)
    correct = (3 / (8 + 1 + 1))
    assert tpr_score == correct


def test_lfpr(pred, truth):
    lfpr_score = lfpr(pred, truth)
    correct = 1 / 3
    assert lfpr_score == correct


def test_ltpr(pred, truth):
    ltpr_score = ltpr(pred, truth)
    correct = 2 / 3
    assert ltpr_score == correct


def test_avd(pred, truth):
    avd_score = avd(pred, truth)
    correct = 0.6
    assert avd_score == correct


def test_corr(pred, truth):
    ps = pred.sum()
    ts = truth.sum()
    eps = 0.1
    pred_vols = [ps, ps + eps, ps - eps]
    truth_vols = [ts, ts + eps, ts - eps]
    corr_score = corr(pred_vols, truth_vols)
    correct = 1.0
    assert pytest.approx(corr_score, 1e-3) == correct


def test_isbi15_score(pred, truth):
    isbi15 = isbi15_score(pred, truth)
    correct = 0.6408730158730158
    assert isbi15 == correct


@pytest.mark.skip('Not implemented.')
def test_assd(pred, truth):
    pass
