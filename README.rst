==============
lesion-metrics
==============


.. image:: https://img.shields.io/pypi/v/lesion_metrics.svg
        :target: https://pypi.python.org/pypi/lesion-metrics

.. image:: https://api.travis-ci.com/jcreinhold/lesion-metrics.svg
        :target: https://travis-ci.com/jcreinhold/lesion-metrics

.. image:: https://readthedocs.org/projects/lesion-metrics/badge/?version=latest
        :target: https://lesion-metrics.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


Various metrics for evaluating lesion segmentations [1]


* Free software: Apache Software License 2.0
* Documentation: https://lesion-metrics.readthedocs.io.

Install
-------

The easiest way to install the package is with::

    pip install lesion-metrics

To install the dependencies of the CLI, install with::

    pip install "lesion-metrics[cli]"

To install the dependencies for total lesion burden/volume computation (see ``lesion_metrics.volume``), install with::

    pip install "lesion-metrics[volume]"

You can also download the source and run::

    python setup.py install

Basic Usage
-----------

You can generate a report of lesion metrics for a directory of predicted labels and truth labels
with the CLI::

    lesion-metrics -p predictions/ -t truth/ -o output.csv

Or you can import the metrics and run them on label images:

.. code-block:: python

    import nibabel as nib
    from lesion_metrics.metrics import dice
    pred = nib.load('pred_label.nii.gz').get_fdata()
    truth = nib.load('truth_label.nii.gz').get_fdata()
    dice_score = dice(pred, truth)


References
----------

[1] Carass, Aaron, et al. "Longitudinal multiple sclerosis lesion segmentation: resource and challenge." NeuroImage 148 (2017): 77-102.
