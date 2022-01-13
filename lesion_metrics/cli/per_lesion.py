"""Console script for lesion_metrics on per-lesion basis."""
import argparse
import builtins
import sys
import typing
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    import medio.image as mioi
    import numpy as np
    import pandas as pd
    import scipy.ndimage
    import skimage.measure

    import lesion_metrics.cli.common as lmcc
    import lesion_metrics.metrics as lmm
    import lesion_metrics.utils as lmu
    import lesion_metrics.volume as lmv


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
        type=lmcc.csv_file_path(),
        required=True,
        help="path to output csv file of results",
    )
    required.add_argument(
        "-p",
        "--pred",
        type=lmcc.dir_or_file_path(),
        default=None,
        help="path to directory of predictions images",
    )
    required.add_argument(
        "-t",
        "--truth",
        type=lmcc.dir_or_file_path(),
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


def main(args: lmcc.ArgType = None) -> builtins.int:
    """Alternate console script for lesion_metrics."""
    if args is None:
        parser = arg_parser()
        args = parser.parse_args()
    elif isinstance(args, list):
        parser = arg_parser()
        args = parser.parse_args(args)
    lmcc.setup_log(args.verbosity)
    pos, dcs, jis, ppvs, tprs, lfdrs, ltprs, avds = [], [], [], [], [], [], [], []
    pred_vols, truth_vols = [], []
    _, pfn, _ = lmcc.split_filename(args.pred)
    _, tfn, _ = lmcc.split_filename(args.truth)
    pfns: typing.List[typing.Optional[builtins.str]] = [pfn]
    tfns: typing.List[typing.Optional[builtins.str]] = [tfn]
    _pred = mioi.Image.from_path(args.pred)
    _truth = mioi.Image.from_path(args.truth)
    pred_vols.append(lmv.SegmentationVolume(_pred).volume())
    truth_vols.append(lmv.SegmentationVolume(_truth).volume())
    pred = _pred.squeeze()
    truth = _truth.squeeze()
    lfdrs.append(lmm.lfdr(pred, truth, iou_threshold=args.iou_threshold))
    ltprs.append(lmm.ltpr(pred, truth, iou_threshold=args.iou_threshold))
    cc_truth, n_truth = skimage.measure.label(truth, return_num=True)
    cc_pred, _ = skimage.measure.label(pred, return_num=True)
    for i in range(1, n_truth + 1):
        truth_lesion_whole_array = cc_truth == i
        position = tuple(
            np.round(scipy.ndimage.center_of_mass(truth_lesion_whole_array), decimals=2)
        )
        intersecting_lesions = set(np.unique(cc_pred[truth_lesion_whole_array])) - {0}
        if intersecting_lesions:
            other_lesions = np.full_like(pred, False)
            for j in intersecting_lesions:
                other_lesions |= cc_pred == j
            lesion_bbox = tuple(lmu.bbox(truth_lesion_whole_array | other_lesions))
            target_lesion = truth_lesion_whole_array[lesion_bbox]
            other_lesion = other_lesions[lesion_bbox]
            _dice = lmm.dice(other_lesion, target_lesion)
            _iou = lmm.jaccard(other_lesion, target_lesion)
            _ppv = lmm.ppv(other_lesion, target_lesion)
            _tpr = lmm.tpr(other_lesion, target_lesion)
            _avd = lmm.avd(other_lesion, target_lesion)
        else:
            _dice = _iou = _ppv = _tpr = _avd = 0.0
        pos.append(position)
        dcs.append(_dice)
        jis.append(_iou)
        ppvs.append(_ppv)
        tprs.append(_tpr)
        avds.append(_avd)
    pfns = lmcc.pad_with_none_to_length(pfns, n_truth)
    tfns = lmcc.pad_with_none_to_length(tfns, n_truth)
    dcs_summary = lmcc.summary_statistics(dcs)
    labels = list(dcs_summary.keys())
    pos.extend(labels)
    dcs.extend(list(dcs_summary.values()))
    jis.extend(list(lmcc.summary_statistics(jis).values()))
    ppvs.extend(list(lmcc.summary_statistics(ppvs).values()))
    tprs.extend(list(lmcc.summary_statistics(tprs).values()))
    avds.extend(list(lmcc.summary_statistics(avds).values()))
    lfdrs = lmcc.pad_with_none_to_length(lfdrs, len(dcs))
    ltprs = lmcc.pad_with_none_to_length(ltprs, len(dcs))
    pred_vols = lmcc.pad_with_none_to_length(pred_vols, len(dcs))
    truth_vols = lmcc.pad_with_none_to_length(truth_vols, len(dcs))
    pfns = lmcc.pad_with_none_to_length(pfns, len(dcs))
    tfns = lmcc.pad_with_none_to_length(tfns, len(dcs))
    shape = lmcc.pad_with_none_to_length([pred.shape], len(dcs))
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
    sys.exit(main())  # pragma: no cover
