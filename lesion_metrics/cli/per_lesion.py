"""Console script for lesion_metrics on per-lesion basis."""
import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, List, Optional, Union

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    import nibabel as nib
    import numpy as np
    import pandas as pd
    from scipy.ndimage import center_of_mass
    from skimage.measure import label

    from lesion_metrics.cli.common import (
        ArgType,
        csv_file_path,
        dir_or_file_path,
        pad_with_none_to_length,
        setup_log,
        split_filename,
        summary_statistics,
    )
    from lesion_metrics.metrics import avd, dice, jaccard, lfdr, ltpr, ppv, tpr
    from lesion_metrics.utils import bbox

    SegmentationVolume: Any
    try:
        import torchio as tio

        from lesion_metrics.volume import SegmentationVolume
    except (ImportError, ModuleNotFoundError):
        SegmentationVolume = None
        tio = None


def arg_parser() -> argparse.ArgumentParser:
    desc = (
        "Calculate a suite of lesion quality metrics "
        "for each lesion in a pair of NIfTI binary (lesion) segmentations."
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
    required.add_argument(
        "-p",
        "--pred",
        type=dir_or_file_path(),
        default=None,
        help="path to directory of predictions images",
    )
    required.add_argument(
        "-t",
        "--truth",
        type=dir_or_file_path(),
        default=None,
        help="path to directory of corresponding truth images",
    )

    options = parser.add_argument_group("Optional")
    options.add_argument(
        "-it",
        "--iou-threshold",
        type=float,
        default=0.0,
        help="iou threshold for detection (in LTPR and LFDR)",
    )
    options.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=0,
        help="increase output verbosity (e.g., -vv is more than -v)",
    )
    return parser


def main(args: ArgType = None) -> int:
    """Alternate console script for lesion_metrics."""
    if args is None:
        parser = arg_parser()
        args = parser.parse_args()
    elif isinstance(args, list):
        parser = arg_parser()
        args = parser.parse_args(args)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    pos, dcs, jis, ppvs, tprs, lfdrs, ltprs, avds = [], [], [], [], [], [], [], []
    pred_vols, truth_vols = [], []
    if SegmentationVolume is None:
        logger.info("Using numpy. Volume is count of positive voxels.")
    else:
        msg = "Using torchio. Volume is count of pos. voxels scaled by resolution."
        logger.info(msg)
    _, pfn, _ = split_filename(args.pred)
    _, tfn, _ = split_filename(args.truth)
    pfns: List[Optional[str]] = [pfn]
    tfns: List[Optional[str]] = [tfn]
    if SegmentationVolume is None:
        pred = nib.load(args.pred).get_fdata() > 0.0
        truth = nib.load(args.truth).get_fdata() > 0.0
        pred_vols.append(pred.sum())
        truth_vols.append(truth.sum())
    else:
        _pred = tio.LabelMap(args.pred)
        _truth = tio.LabelMap(args.truth)
        pred_vols.append(SegmentationVolume(_pred).volume())
        truth_vols.append(SegmentationVolume(_truth).volume())
        pred = _pred.numpy().squeeze()
        truth = _truth.numpy().squeeze()
    lfdrs.append(lfdr(pred, truth, iou_threshold=args.iou_threshold))
    ltprs.append(ltpr(pred, truth, iou_threshold=args.iou_threshold))
    cc_truth, n_truth = label(truth, return_num=True)
    cc_pred, _ = label(pred, return_num=True)
    for i in range(1, n_truth + 1):
        truth_lesion_whole_array = cc_truth == i
        position = tuple(np.round(center_of_mass(truth_lesion_whole_array), decimals=2))
        intersecting_lesions = set(np.unique(cc_pred[truth_lesion_whole_array])) - {0}
        if intersecting_lesions:
            other_lesions = np.full_like(pred, False)
            for j in intersecting_lesions:
                other_lesions |= cc_pred == j
            lesion_bbox = tuple(bbox(truth_lesion_whole_array | other_lesions))
            target_lesion = truth_lesion_whole_array[lesion_bbox]
            other_lesion = other_lesions[lesion_bbox]
            _dice = dice(other_lesion, target_lesion)
            _iou = jaccard(other_lesion, target_lesion)
            _ppv = ppv(other_lesion, target_lesion)
            _tpr = tpr(other_lesion, target_lesion)
            _avd = avd(other_lesion, target_lesion)
        else:
            _dice = _iou = _ppv = _tpr = _avd = 0.0
        pos.append(position)
        dcs.append(_dice)
        jis.append(_iou)
        ppvs.append(_ppv)
        tprs.append(_tpr)
        avds.append(_avd)
    pfns = pad_with_none_to_length(pfns, n_truth)
    tfns = pad_with_none_to_length(tfns, n_truth)
    dcs_summary = summary_statistics(dcs)
    labels = list(dcs_summary.keys())
    pos.extend(labels)
    dcs.extend(list(dcs_summary.values()))
    jis.extend(list(summary_statistics(jis).values()))
    ppvs.extend(list(summary_statistics(ppvs).values()))
    tprs.extend(list(summary_statistics(tprs).values()))
    avds.extend(list(summary_statistics(avds).values()))
    lfdrs = pad_with_none_to_length(lfdrs, len(dcs))
    ltprs = pad_with_none_to_length(ltprs, len(dcs))
    pred_vols = pad_with_none_to_length(pred_vols, len(dcs))
    truth_vols = pad_with_none_to_length(truth_vols, len(dcs))
    pfns = pad_with_none_to_length(pfns, len(dcs))
    tfns = pad_with_none_to_length(tfns, len(dcs))
    shape = pad_with_none_to_length([pred.shape], len(dcs))
    out = {
        "Pred": pfns,
        "Truth": tfns,
        "Center": pos,
        "Dice": dcs,
        "Jaccard": jis,
        "PPV": ppvs,
        "TPR": tprs,
        "AVD": avds,
        "LFDR": lfdrs,
        "LTPR": ltprs,
        "Pred. Vol.": pred_vols,
        "Truth. Vol.": truth_vols,
        "Shape": shape,
    }
    pd.DataFrame(out).to_csv(args.out_file, index=False)
    return 0


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    sys.exit(main(args))  # pragma: no cover
