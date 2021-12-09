# -*- coding: utf-8 -*-
"""
lesion_metrics.cli.common

common cli functions

Author: Jacob Reinhold (jcreinhold@gmail.com)
Created on: 05 Dec 2021
"""

__all__ = [
    "ArgType",
    "check_files",
    "csv_file_path",
    "dir_path",
    "dir_or_file_path",
    "file_path",
    "glob_imgs",
    "pad_with_none_to_length",
    "setup_log",
    "split_filename",
    "summary_statistics",
]

import argparse
import logging
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

ArgType = Optional[Union[argparse.Namespace, List[str]]]


def split_filename(
    filepath: Union[str, Path], *, resolve: bool = False
) -> Tuple[Path, str, str]:
    """split a filepath into the directory, base, and extension"""
    filepath = Path(filepath)
    if resolve:
        filepath.resolve()
    path = filepath.parent
    _base = Path(filepath.stem)
    ext = filepath.suffix
    if ext == ".gz":
        ext2 = _base.suffix
        base = str(_base.stem)
        ext = ext2 + ext
    else:
        base = str(_base)
    return Path(path), base, ext


def setup_log(verbosity: int) -> None:
    """get logger with appropriate logging level and message"""
    if verbosity == 1:
        level = logging.getLevelName("INFO")
    elif verbosity >= 2:
        level = logging.getLevelName("DEBUG")
    else:
        level = logging.getLevelName("WARNING")
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt, level=level)
    logging.captureWarnings(True)


class _ParseType:
    @property
    def __name__(self) -> str:
        name = self.__class__.__name__
        assert isinstance(name, str)
        return name

    def __str__(self) -> str:
        return self.__name__


class file_path(_ParseType):
    def __call__(self, string: str) -> Path:
        path = Path(string)
        if not path.is_file():
            msg = f"{string} is not a valid path to a file."
            raise argparse.ArgumentTypeError(msg)
        return path


class csv_file_path(_ParseType):
    def __call__(self, string: str) -> Path:
        if not string.endswith(".csv") or not string.isprintable():
            msg = (
                f"{string} is not a valid path to a csv file.\n"
                "file needs to end with csv and only contain "
                "printable characters."
            )
            raise argparse.ArgumentTypeError(msg)
        path = Path(string)
        return path


class dir_path(_ParseType):
    def __call__(self, string: str) -> Path:
        path = Path(string)
        if not path.is_dir():
            msg = f"{string} is not a valid path to a directory."
            raise argparse.ArgumentTypeError(msg)
        return path


class dir_or_file_path(_ParseType):
    def __call__(self, string: str) -> Path:
        path = Path(string)
        if not path.is_dir() and not path.is_file():
            msg = f"{string} is not a valid path to a directory or file."
            raise argparse.ArgumentTypeError(msg)
        return path


def glob_imgs(path: Path, ext: str = "*.nii*") -> List[Path]:
    """grab all `ext` files in a directory and sort them for consistency"""
    return sorted(path.glob(ext))


def check_files(*files: Path) -> None:
    msg = ""
    for f in files:
        if not f.is_file():
            msg += f"{f} is not a valid path.\n"
    if msg:
        raise ValueError(msg + "Aborting.")


def summary_statistics(data: Union[List[int], List[float]]) -> OrderedDict:
    funcs: OrderedDict[str, Callable] = OrderedDict()
    funcs["Avg"] = np.mean
    funcs["Std"] = np.std
    funcs["Min"] = np.min
    funcs["25%"] = partial(np.percentile, q=25.0)
    funcs["50%"] = np.median
    funcs["75%"] = partial(np.percentile, q=75.0)
    funcs["Max"] = np.max
    return OrderedDict((label, f(data)) for label, f in funcs.items())


def pad_with_none_to_length(lst: List[Any], length: int) -> List[Any]:
    current_length = len(lst)
    if length <= current_length:
        return lst
    n = length - current_length
    padded = lst + ([None] * n)
    assert len(padded) == length
    return padded
