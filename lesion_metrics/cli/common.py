"""Common cli functions
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
import builtins
import collections
import functools
import logging
import os
import pathlib
import typing

import numpy as np

ArgType = typing.Optional[typing.Union[argparse.Namespace, typing.List[builtins.str]]]


def split_filename(
    filepath: typing.Union[os.PathLike, builtins.str], *, resolve: builtins.bool = False
) -> typing.Tuple[pathlib.Path, builtins.str, builtins.str]:
    """split a filepath into the directory, base, and extension"""
    filepath = pathlib.Path(filepath)
    if resolve:
        filepath.resolve()
    path = filepath.parent
    _base = pathlib.Path(filepath.stem)
    ext = filepath.suffix
    if ext == ".gz":
        ext2 = _base.suffix
        base = str(_base.stem)
        ext = ext2 + ext
    else:
        base = str(_base)
    return pathlib.Path(path), base, ext


def setup_log(verbosity: builtins.int) -> None:
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
    def __name__(self) -> builtins.str:
        name = self.__class__.__name__
        assert isinstance(name, str)
        return name

    def __str__(self) -> str:
        return self.__name__


class file_path(_ParseType):
    def __call__(self, string: builtins.str) -> pathlib.Path:
        path = pathlib.Path(string)
        if not path.is_file():
            msg = f"{string} is not a valid path to a file."
            raise argparse.ArgumentTypeError(msg)
        return path


class csv_file_path(_ParseType):
    def __call__(self, string: builtins.str) -> pathlib.Path:
        if not string.endswith(".csv") or not string.isprintable():
            msg = (
                f"{string} is not a valid path to a csv file.\n"
                "file needs to end with csv and only contain "
                "printable characters."
            )
            raise argparse.ArgumentTypeError(msg)
        path = pathlib.Path(string)
        return path


class dir_path(_ParseType):
    def __call__(self, string: builtins.str) -> pathlib.Path:
        path = pathlib.Path(string)
        if not path.is_dir():
            msg = f"{string} is not a valid path to a directory."
            raise argparse.ArgumentTypeError(msg)
        return path


class dir_or_file_path(_ParseType):
    def __call__(self, string: builtins.str) -> pathlib.Path:
        path = pathlib.Path(string)
        if not path.is_dir() and not path.is_file():
            msg = f"{string} is not a valid path to a directory or file."
            raise argparse.ArgumentTypeError(msg)
        return path


def glob_imgs(
    path: pathlib.Path, ext: builtins.str = "*.nii*"
) -> typing.List[pathlib.Path]:
    """grab all `ext` files in a directory and sort them for consistency"""
    return sorted(path.glob(ext))


def check_files(*files: pathlib.Path) -> None:
    msg = ""
    for f in files:
        if not f.is_file():
            msg += f"{f} is not a valid path.\n"
    if msg:
        raise ValueError(msg + "Aborting.")


def summary_statistics(
    data: typing.Sequence[builtins.float],
) -> collections.OrderedDict:
    funcs: collections.OrderedDict[builtins.str, typing.Callable]
    funcs = collections.OrderedDict()
    funcs["Avg"] = np.mean
    funcs["Std"] = np.std
    funcs["Min"] = np.min
    funcs["25%"] = functools.partial(np.percentile, q=25.0)
    funcs["50%"] = np.median
    funcs["75%"] = functools.partial(np.percentile, q=75.0)
    funcs["Max"] = np.max
    return collections.OrderedDict((label, f(data)) for label, f in funcs.items())


def pad_with_none_to_length(
    lst: typing.List[typing.Any], length: builtins.int
) -> typing.List[typing.Any]:
    current_length = len(lst)
    if length <= current_length:
        return lst
    n = length - current_length
    padded = lst + ([None] * n)
    assert len(padded) == length
    return padded
