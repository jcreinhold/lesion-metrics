"""Console script for lesion_metrics."""
import argparse
from glob import glob
import logging
import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    import nibabel as nib
    import pandas as pd
    from lesion_metrics import *


def setup_log(verbosity: int):
    """ get logger with appropriate logging level and message """
    if verbosity == 1:
        level = logging.getLevelName('INFO')
    elif verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)


def arg_parser():
    parser = argparse.ArgumentParser(description='Calculate a suite of lesion quality metrics '
                                                 'for a set of NIfTI binary (lesion) segmentations.')

    required = parser.add_argument_group('Required')
    required.add_argument('-p', '--pred-dir', type=str, required=True,
                          help='path to directory of predictions images')
    required.add_argument('-t', '--truth-dir', type=str, required=True,
                          help='path to directory of corresponding truth images')
    required.add_argument('-o', '--out-file', type=str, required=True,
                          help='path to output csv file of results')

    options = parser.add_argument_group('Optional')
    options.add_argument('-c', '--output-correlation', action="store_true",
                         help="output the volume correlation of the set of images")
    options.add_argument('-v', '--verbosity', action="count", default=0,
                         help="increase output verbosity (e.g., -vv is more than -v)")
    return parser


def glob_imgs(path, ext='*.nii*'):
    """ grab all `ext` files in a directory and sort them for consistency """
    fns = sorted(glob(os.path.join(path, ext)))
    return fns


def split_filename(filepath):
    """ split a filepath into the directory, base, and extension """
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext


def main(args=None):
    """Console script for lesion_metrics."""
    if args is None:
        parser = arg_parser()
        args = parser.parse_args()
    elif isinstance(args, list):
        parser = arg_parser()
        args = parser.parse_args(args)
    setup_log(args.verbosity)
    logger = logging.getLogger(__name__)
    pred_fns = glob_imgs(args.pred_dir)
    n_pred = len(pred_fns)
    truth_fns = glob_imgs(args.truth_dir)
    n_truth = len(truth_fns)
    if n_pred != n_truth or n_pred == 0:
        raise ValueError(f'Number of prediction and truth images must be equal and non-zero '
                         f'(# Pred.={n_pred}; # Truth={n_truth})')
    if args.output_correlation and n_pred == 1:
        raise ValueError('If --output-correlation enabled, the input '
                         'directories must contain more than 1 image.')
    dcs, jis, ppvs, tprs, lfprs, ltprs, avds, isbis = [], [], [], [], [], [], [], []
    pfns, tfns = [], []
    pred_vols, truth_vols = [], []
    for pf, tf in zip(pred_fns, truth_fns):
        _, pfn, _ = split_filename(pf)
        _, tfn, _ = split_filename(tf)
        pfns.append(pfn)
        tfns.append(tfn)
        pred = (nib.load(pf).get_fdata() > 0)
        truth = (nib.load(tf).get_fdata() > 0)
        if args.output_correlation:
            pred_vols.append(pred)
            truth_vols.append(truth)
        dcs.append(dice(pred, truth))
        jis.append(jaccard(pred, truth))
        ppvs.append(ppv(pred, truth))
        tprs.append(tpr(pred, truth))
        lfprs.append(lfpr(pred, truth))
        ltprs.append(ltpr(pred, truth))
        avds.append(avd(pred, truth))
        isbis.append(isbi15_score(pred, truth))
        logger.info(f'Pred: {pfn}; Truth: {tfn}; Dice: {dcs[-1]:0.2f}; Jacc: {jis[-1]:0.2f}; '
                    f'PPV: {ppvs[-1]:0.2f}; TPR: {tprs[-1]:0.2f}; LFPR: {lfprs[-1]:0.2f}; '
                    f'LTPR: {ltprs[-1]:0.2f}; AVD: {avds[-1]:0.2f}; ISBI15 Score: {isbis[-1]:0.2f}')
    out = {'Pred': pfns,
           'Truth': tfns,
           'Dice': dcs,
           'Jaccard': jis,
           'PPV': ppvs,
           'TPR': tprs,
           'LFPR': lfprs,
           'LTPR': ltprs,
           'AVD': avds,
           'ISBI15 Score': isbis}
    if args.output_correlation:
        c = corr(pred_vols, truth_vols)
        logger.info(f'Volume correlation: {c:0.2f}')
        out['Correlation'] = [None] * n_pred
        out['Correlation'][0] = c
    pd.DataFrame(out).to_csv(args.out_file)
    return 0


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    sys.exit(main(args))  # pragma: no cover
