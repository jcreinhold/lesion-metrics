#!/usr/bin/env python
"""Tests for `lesion_metrics` package."""

import builtins
import os
import pathlib
import shutil
import typing

import pytest

import lesion_metrics.cli.aggregate as lmca


@pytest.fixture
def cwd() -> pathlib.Path:
    cwd = pathlib.Path.cwd().resolve()
    if cwd.name == "tests":
        return cwd
    cwd = (cwd / "tests").resolve(strict=True)
    return cwd


@pytest.fixture(scope="session")
def temp_dir(tmpdir_factory) -> pathlib.Path:  # type: ignore[no-untyped-def]
    return pathlib.Path(tmpdir_factory.mktemp("out"))


def _copy_test_data(
    cwd: pathlib.Path,
    temp_dir: pathlib.Path,
    name: typing.Union[builtins.str, os.PathLike],
) -> pathlib.Path:
    fn = cwd / "test_data" / name / f"{name}.nii.gz"
    tmp = temp_dir / name
    os.mkdir(tmp)
    shutil.copyfile(fn, tmp / f"{name}1.nii.gz")
    shutil.copyfile(fn, tmp / f"{name}2.nii.gz")
    return tmp


@pytest.fixture
def pred_dir(cwd: pathlib.Path, temp_dir: pathlib.Path) -> pathlib.Path:
    return _copy_test_data(cwd, temp_dir, "pred")


@pytest.fixture
def truth_dir(cwd: pathlib.Path, temp_dir: pathlib.Path) -> pathlib.Path:
    return _copy_test_data(cwd, temp_dir, "truth")


@pytest.fixture
def out_file(temp_dir: pathlib.Path) -> pathlib.Path:
    return temp_dir / "test.csv"


@pytest.fixture
def args(
    pred_dir: pathlib.Path, truth_dir: pathlib.Path, out_file: pathlib.Path
) -> typing.List[builtins.str]:
    return f"-p {pred_dir} -t {truth_dir} -o {out_file} -c".split()


def test_cli(args: typing.List[builtins.str]) -> None:
    retval = lmca.main(args)
    assert retval == 0
