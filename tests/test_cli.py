#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for `lesion_metrics` package."""

import os
from pathlib import Path
import shutil

import pytest

from lesion_metrics.cli import main


@pytest.fixture
def file():
    return Path(__file__).resolve()


@pytest.fixture
def cwd(file):
    return file.parent


@pytest.fixture(scope='session')
def temp_dir(tmpdir_factory):
    return Path(tmpdir_factory.mktemp('out'))


def _copy_test_data(cwd, temp_dir, name):
    fn = cwd / 'test_data' / name / f'{name}.nii.gz'
    tmp = temp_dir / name
    os.mkdir(tmp)
    shutil.copyfile(fn, tmp / f'{name}1.nii.gz')
    shutil.copyfile(fn, tmp / f'{name}2.nii.gz')
    return tmp


@pytest.fixture
def pred_dir(cwd, temp_dir):
    return _copy_test_data(cwd, temp_dir, 'pred')


@pytest.fixture
def truth_dir(cwd, temp_dir):
    return _copy_test_data(cwd, temp_dir, 'truth')


@pytest.fixture
def out_file(temp_dir):
    return temp_dir / 'test.csv'


@pytest.fixture
def args(pred_dir, truth_dir, out_file):
    return f'-p {pred_dir} -t {truth_dir} -o {out_file} -c'.split()


def test_cli(args):
    retval = main(args)
    assert retval == 0
