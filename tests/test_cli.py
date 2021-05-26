#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `lesion_metrics` package."""

import os
from pathlib import Path
import shutil
from typing import List, Union

import pytest

from lesion_metrics.cli import main


@pytest.fixture
def file() -> Path:
    return Path(__file__).resolve()


@pytest.fixture
def cwd(file: Path) -> Path:
    return file.parent


@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory) -> Path:
    return Path(tmpdir_factory.mktemp("out"))


def _copy_test_data(cwd: Path, temp_dir: Path, name: Union[str, os.PathLike]) -> Path:
    fn = cwd / "test_data" / name / f"{name}.nii.gz"
    tmp = temp_dir / name
    os.mkdir(tmp)
    shutil.copyfile(fn, tmp / f"{name}1.nii.gz")
    shutil.copyfile(fn, tmp / f"{name}2.nii.gz")
    return tmp


@pytest.fixture
def pred_dir(cwd: Path, temp_dir: Path) -> Path:
    return _copy_test_data(cwd, temp_dir, "pred")


@pytest.fixture
def truth_dir(cwd: Path, temp_dir: Path) -> Path:
    return _copy_test_data(cwd, temp_dir, "truth")


@pytest.fixture
def out_file(temp_dir: Path) -> Path:
    return temp_dir / "test.csv"


@pytest.fixture
def args(pred_dir: Path, truth_dir: Path, out_file: Path) -> List[str]:
    return f"-p {pred_dir} -t {truth_dir} -o {out_file} -c".split()


def test_cli(args: List[str]):
    retval = main(args)
    assert retval == 0
