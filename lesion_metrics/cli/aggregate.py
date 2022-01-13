"""Console script for lesion_metrics."""
import argparse
import builtins
import logging
import pathlib
import sys
import typing
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    import medio.image as mioi
    import pandas as pd

    import lesion_metrics.cli.common as lmcc
    import lesion_metrics.metrics as lmm
    import lesion_metrics.volume as lmv


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
        type=lmcc.csv_file_path(),
        required=True,
        help="path to output csv file of results",
    )

    primary = parser.add_argument_group("Primary Input (provide these instead of -f)")
    primary.add_argument(
        "-p",
        "--pred-dir",
        type=lmcc.dir_path(),
        default=None,
        help="path to directory of predictions images",
    )
    primary.add_argument(
        "-t",
        "--truth-dir",
        type=lmcc.dir_path(),
        default=None,
        help="path to directory of corresponding truth images",
    )

    alt = parser.add_argument_group(
        "Alternative Input (provide this instead of -p & -t)"
    )
    alt.add_argument(
        "-f",
        "--in-file",
        type=lmcc.file_path(),
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
    """Console script for lesion_metrics."""
    if args is None:
        parser = arg_parser()
        args = parser.parse_args()
    elif isinstance(args, list):
        parser = arg_parser()
        args = parser.parse_args(args)
    lmcc.setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    use_dirs = args.pred_dir is not None and args.truth_dir is not None
    use_csv = args.in_file is not None
    if use_dirs and not use_csv:
        pred_fns = lmcc.glob_imgs(args.pred_dir)
        truth_fns = lmcc.glob_imgs(args.truth_dir)
    elif use_csv and not use_dirs:
        csv = pd.read_csv(args.in_file)
        pred_fns = [pathlib.Path(f) for f in csv["pred"].to_list()]
        truth_fns = [pathlib.Path(f) for f in csv["truth"].to_list()]
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
    dcs, jis, ppvs, tprs, lfdrs, ltprs, avds, isbis = [], [], [], [], [], [], [], []
    pfns: typing.List[typing.Optional[builtins.str]] = []
    tfns: typing.List[typing.Optional[builtins.str]] = []
    pred_vols, truth_vols = [], []
    pred_counts, truth_counts = [], []
    for pf, tf in zip(pred_fns, truth_fns):
        lmcc.check_files(pf, tf)
        _, pfn, _ = lmcc.split_filename(pf)
        _, tfn, _ = lmcc.split_filename(tf)
        pfns.append(pfn)
        tfns.append(tfn)
        _pred = mioi.Image.from_path(pf)
        _truth = mioi.Image.from_path(tf)
        pred_vols.append(lmv.SegmentationVolume(_pred).volume())
        truth_vols.append(lmv.SegmentationVolume(_truth).volume())
        pred = _pred.squeeze()
        truth = _truth.squeeze()
        dcs.append(lmm.dice(pred, truth))
        jis.append(lmm.jaccard(pred, truth))
        ppvs.append(lmm.ppv(pred, truth))
        tprs.append(lmm.tpr(pred, truth))
        __lfdr = lmm.lfdr(
            pred, truth, iou_threshold=args.iou_threshold, return_pred_count=True
        )
        assert isinstance(__lfdr, tuple)
        _lfdr, n_pred = __lfdr
        lfdrs.append(_lfdr)
        __ltpr = lmm.ltpr(
            pred, truth, iou_threshold=args.iou_threshold, return_truth_count=True
        )
        assert isinstance(__ltpr, tuple)
        _ltpr, n_truth = __ltpr
        ltprs.append(_ltpr)
        avds.append(lmm.avd(pred, truth))
        isbi15_score = lmm.isbi15_score_from_metrics(
            dcs[-1], ppvs[-1], lfdrs[-1], ltprs[-1]
        )
        isbis.append(isbi15_score)
        pred_counts.append(n_pred)
        truth_counts.append(n_truth)
        logger.info(
            f"Pred: {pfn}; Truth: {tfn}; Dice: {dcs[-1]:0.2f}; Jacc: {jis[-1]:0.2f}; "
            f"PPV: {ppvs[-1]:0.2f}; TPR: {tprs[-1]:0.2f}; LFDR: {lfdrs[-1]:0.2f}; "
            f"LTPR: {ltprs[-1]:0.2f}; AVD: {avds[-1]:0.2f}; ISBI15: {isbis[-1]:0.2f}"
        )
    dcs_summary = lmcc.summary_statistics(dcs)
    labels = list(dcs_summary.keys())
    tfns.extend(labels)
    pfns = lmcc.pad_with_none_to_length(pfns, len(tfns))
    dcs.extend(list(dcs_summary.values()))
    jis.extend(list(lmcc.summary_statistics(jis).values()))
    ppvs.extend(list(lmcc.summary_statistics(ppvs).values()))
    tprs.extend(list(lmcc.summary_statistics(tprs).values()))
    lfdrs.extend(list(lmcc.summary_statistics(lfdrs).values()))
    ltprs.extend(list(lmcc.summary_statistics(ltprs).values()))
    avds.extend(list(lmcc.summary_statistics(avds).values()))
    isbis.extend(list(lmcc.summary_statistics(isbis).values()))
    pred_vols.extend(list(lmcc.summary_statistics(pred_vols).values()))
    truth_vols.extend(list(lmcc.summary_statistics(truth_vols).values()))
    pred_counts.extend(list(lmcc.summary_statistics(pred_counts).values()))
    truth_counts.extend(list(lmcc.summary_statistics(truth_counts).values()))
    out = {
        "Pred": pfns,
        "Truth": tfns,
        "Dice": dcs,
        "Jaccard": jis,
        "PPV": ppvs,
        "TPR": tprs,
        "LFDR": lfdrs,
        "LTPR": ltprs,
        "AVD": avds,
        "ISBI15 Score": isbis,
        "Pred. Vol.": pred_vols,
        "Truth. Vol.": truth_vols,
        "Pred. Count": pred_counts,
        "Truth. Count": truth_counts,
    }
    if args.output_correlation:
        vc = lmm.corr(pred_vols, truth_vols)
        logger.info(f"Volume correlation: {vc:0.2f}")
        out["Vol. Correlation"] = lmcc.pad_with_none_to_length([vc], len(truth_vols))
        cc = lmm.corr(pred_counts, truth_counts)
        logger.info(f"Count correlation: {cc:0.2f}")
        out["Count Correlation"] = lmcc.pad_with_none_to_length([cc], len(truth_vols))
    pd.DataFrame(out).to_csv(args.out_file, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
