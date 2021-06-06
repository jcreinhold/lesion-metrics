"""Console script for lesion_metrics."""
import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    import nibabel as nib
    import pandas as pd
    from lesion_metrics.metrics import (
        dice,
        jaccard,
        ppv,
        tpr,
        lfpr,
        ltpr,
        avd,
        corr,
        isbi15_score,
    )

ArgType = Optional[Union[argparse.Namespace, List[str]]]


def split_filename(filepath: Union[str, Path]) -> Tuple[Path, str, str]:
    """ split a filepath into the directory, base, and extension """
    filepath = Path(filepath).resolve()
    path = filepath.parent
    base = Path(filepath.stem)
    ext = filepath.suffix
    if ext == ".gz":
        ext2 = base.suffix
        base = base.stem
        ext = ext2 + ext
    return Path(path), str(base), ext


def setup_log(verbosity: int):
    """ get logger with appropriate logging level and message """
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
        return self.__class__.__name__

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


def arg_parser() -> argparse.ArgumentParser:
    desc = (
        "Calculate a suite of lesion quality metrics "
        "for a set of NIfTI binary (lesion) segmentations."
    )
    parser = argparse.ArgumentParser(description=desc)

    required = parser.add_argument_group("Required")
    required.add_argument(
        "-o",
        "--out-file",
        type=csv_file_path(),
        required=True,
        help="path to output csv file of results",
    )

    primary = parser.add_argument_group("Primary Input (provide these instead of -f)")
    primary.add_argument(
        "-p",
        "--pred-dir",
        type=dir_path(),
        default=None,
        help="path to directory of predictions images",
    )
    primary.add_argument(
        "-t",
        "--truth-dir",
        type=dir_path(),
        default=None,
        help="path to directory of corresponding truth images",
    )

    alt = parser.add_argument_group(
        "Alternative Input (provide this instead of -p & -t)"
    )
    alt.add_argument(
        "-f",
        "--in-file",
        type=file_path(),
        default=None,
        help=(
            "path to input csv file with (at least) two columns named "
            "`pred` and `truth` consisting of paths to prediction and "
            "corresponding truth images"
        ),
    )

    options = parser.add_argument_group("Optional")
    options.add_argument(
        "-c",
        "--output-correlation",
        action="store_true",
        help="output the volume correlation of the set of images",
    )
    options.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity (e.g., -vv is more than -v)",
    )
    return parser


def glob_imgs(path: Path, ext: str = "*.nii*") -> List[Path]:
    """ grab all `ext` files in a directory and sort them for consistency """
    return sorted(path.glob(ext))


def _check_files(*files: Path):
    msg = ""
    for f in files:
        if not f.is_file():
            msg += f"{f} is not a valid path.\n"
    if msg:
        raise ValueError(msg + "Aborting.")


def main(args: ArgType = None) -> int:
    """Console script for lesion_metrics."""
    if args is None:
        parser = arg_parser()
        args = parser.parse_args()
    elif isinstance(args, list):
        parser = arg_parser()
        args = parser.parse_args(args)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    use_dirs = args.pred_dir is not None and args.truth_dir is not None
    use_csv = args.in_file is not None
    if use_dirs and not use_csv:
        pred_fns = glob_imgs(args.pred_dir)
        truth_fns = glob_imgs(args.truth_dir)
    elif use_csv and not use_dirs:
        csv = pd.read_csv(args.in_file)
        pred_fns = csv["pred"].to_list()
        truth_fns = csv["truth"].to_list()
    else:
        raise ValueError(
            "Only (`--pred-dir` AND `--truth-dir`) OR "
            "`--in-file` should be provided."
        )
    n_pred = len(pred_fns)
    n_truth = len(truth_fns)
    if n_pred != n_truth or n_pred == 0:
        raise ValueError(
            f"Number of prediction and truth images must be equal and non-zero "
            f"(# Pred.={n_pred}; # Truth={n_truth})"
        )
    if args.output_correlation and n_pred == 1:
        raise ValueError(
            "If --output-correlation enabled, the input "
            "directories must contain more than 1 image."
        )
    dcs, jis, ppvs, tprs, lfprs, ltprs, avds, isbis = [], [], [], [], [], [], [], []
    pfns, tfns = [], []
    pred_vols, truth_vols = [], []
    for pf, tf in zip(pred_fns, truth_fns):
        _check_files(pf, tf)
        _, pfn, _ = split_filename(pf)
        _, tfn, _ = split_filename(tf)
        pfns.append(pfn)
        tfns.append(tfn)
        pred = nib.load(pf).get_fdata() > 0.0
        truth = nib.load(tf).get_fdata() > 0.0
        if args.output_correlation:
            pred_vols.append(pred.sum())
            truth_vols.append(truth.sum())
        dcs.append(dice(pred, truth))
        jis.append(jaccard(pred, truth))
        ppvs.append(ppv(pred, truth))
        tprs.append(tpr(pred, truth))
        lfprs.append(lfpr(pred, truth))
        ltprs.append(ltpr(pred, truth))
        avds.append(avd(pred, truth))
        isbis.append(isbi15_score(pred, truth))
        logger.info(
            f"Pred: {pfn}; Truth: {tfn}; Dice: {dcs[-1]:0.2f}; Jacc: {jis[-1]:0.2f}; "
            f"PPV: {ppvs[-1]:0.2f}; TPR: {tprs[-1]:0.2f}; LFPR: {lfprs[-1]:0.2f}; "
            f"LTPR: {ltprs[-1]:0.2f}; AVD: {avds[-1]:0.2f}; ISBI15: {isbis[-1]:0.2f}"
        )
    out = {
        "Pred": pfns,
        "Truth": tfns,
        "Dice": dcs,
        "Jaccard": jis,
        "PPV": ppvs,
        "TPR": tprs,
        "LFPR": lfprs,
        "LTPR": ltprs,
        "AVD": avds,
        "ISBI15 Score": isbis,
    }
    if args.output_correlation:
        c = corr(pred_vols, truth_vols)
        logger.info(f"Volume correlation: {c:0.2f}")
        out["Correlation"] = [None] * n_pred
        out["Correlation"][0] = c
    pd.DataFrame(out).to_csv(args.out_file)
    return 0


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    sys.exit(main(args))  # pragma: no cover
