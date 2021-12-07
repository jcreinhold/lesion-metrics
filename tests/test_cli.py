#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `lesion_metrics` package."""

import os
import shutil
from pathlib import Path
from typing import List, Union

import pytest

from lesion_metrics.cli.aggregate import main


@pytest.fixture
def cwd() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "tests":
        return cwd
    cwd = (cwd / "tests").resolve(strict=True)
    return cwd


@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory) -> Path:  # type: ignore[no-untyped-def]
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


def test_cli(args: List[str]) -> None:
    retval = main(args)
    assert retval == 0
