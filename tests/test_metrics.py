#!/usr/bin/env python
"""Tests for `lesion_metrics` package."""

import builtins
import pathlib

import medio.image as mioi
import pytest

import lesion_metrics.metrics as lmm
import lesion_metrics.typing as lmt
import lesion_metrics.volume as lmv

backends = ["numpy"]
try:
    import torch

    backends.append("torch")
except (ModuleNotFoundError, ImportError):
    pass


@pytest.fixture
def cwd() -> pathlib.Path:
    cwd = pathlib.Path.cwd().resolve()
    if cwd.name == "tests":
        return cwd
    cwd = (cwd / "tests").resolve(strict=True)
    return cwd


@pytest.fixture(params=backends)
def backend(request) -> builtins.str:  # type: ignore[no-untyped-def]
    _backend: str = request.param
    return _backend


@pytest.fixture
def pred_filename(cwd: pathlib.Path) -> pathlib.Path:
    return cwd / "test_data" / "pred" / "pred.nii.gz"


@pytest.fixture
def pred(backend: builtins.str, pred_filename: pathlib.Path) -> lmt.Label:
    pred_data: lmt.Label = mioi.Image.from_path(pred_filename)
    if backend == "torch":
        pred_data = torch.from_numpy(pred_data)  # type: ignore[assignment]
    return pred_data


@pytest.fixture
def truth_filename(cwd: pathlib.Path) -> pathlib.Path:
    return cwd / "test_data" / "truth" / "truth.nii.gz"


@pytest.fixture
def truth(backend: builtins.str, truth_filename: pathlib.Path) -> lmt.Label:
    truth_data: lmt.Label = mioi.Image.from_path(truth_filename)
    if backend == "torch":
        truth_data = torch.from_numpy(truth_data)  # type: ignore[assignment]
    return truth_data


def test_dice(pred: lmt.Label, truth: lmt.Label) -> None:
    dice_coef = lmm.dice(pred, truth)
    correct = 2 * (3 / ((8 + 1 + 1) + (2 + 1 + 1)))
    assert dice_coef == correct


def test_jaccard(pred: lmt.Label, truth: lmt.Label) -> None:
    jaccard_idx = lmm.jaccard(pred, truth)
    correct = 3 / ((8 + 1 + 1) + 1)
    assert jaccard_idx == correct


def test_ppv(pred: lmt.Label, truth: lmt.Label) -> None:
    ppv_score = lmm.ppv(pred, truth)
    correct = 3 / (2 + 1 + 1)
    assert ppv_score == correct


def test_tpr(pred: lmt.Label, truth: lmt.Label) -> None:
    tpr_score = lmm.tpr(pred, truth)
    correct = 3 / (8 + 1 + 1)
    assert tpr_score == correct


def test_lfdr(pred: lmt.Label, truth: lmt.Label) -> None:
    lfpr_score = lmm.lfdr(pred, truth)
    correct = 1 / 3
    assert lfpr_score == correct


def test_ltpr(pred: lmt.Label, truth: lmt.Label) -> None:
    ltpr_score = lmm.ltpr(pred, truth)
    correct = 2 / 3
    assert ltpr_score == correct


def test_avd(pred: lmt.Label, truth: lmt.Label) -> None:
    avd_score = lmm.avd(pred, truth)
    correct = 0.6
    assert avd_score == correct


def test_corr(pred: lmt.Label, truth: lmt.Label) -> None:
    ps = pred.sum()
    ts = truth.sum()
    eps = 0.1
    pred_vols = [ps, ps + eps, ps - eps]
    truth_vols = [ts, ts + eps, ts - eps]
    corr_score = lmm.corr(pred_vols, truth_vols)
    correct = 1.0
    assert pytest.approx(corr_score, 1e-3) == correct


def test_isbi15_score(pred: lmt.Label, truth: lmt.Label) -> None:
    isbi15 = lmm.isbi15_score(pred, truth)
    correct = 0.6408730158730158
    assert isbi15 == pytest.approx(correct, 1e-3)


def test_segmentation_volume(pred_filename: pathlib.Path) -> None:
    sv = lmv.SegmentationVolume.from_filename(pred_filename)
    vol = sv.volume()
    assert vol == 4.0


@pytest.mark.skip("Not implemented.")
def test_assd(pred: lmt.Label, truth: lmt.Label) -> None:
    pass
