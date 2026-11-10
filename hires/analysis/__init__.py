"""Trait-agreement plots comparing ground truth and predictions.

.. note::
   This module is legacy code that is scheduled for removal. It needs the
   ``analysis`` extra: ``pip install 'hireseg[analysis]'``.

The functions take the table returned by
``hires.metrics.match(gt, pred).to_frame()``.
"""

from .analyis import (
    available_descriptor_columns,
    bias,
    bland_altman,
    distributions,
    per_sample_abundance,
    plot_per_sample_abundance,
    target_diagram,
    target_plot,
    taylor_plot,
)

__all__ = [
    "available_descriptor_columns",
    "bias",
    "bland_altman",
    "distributions",
    "per_sample_abundance",
    "plot_per_sample_abundance",
    "target_diagram",
    "target_plot",
    "taylor_plot",
]
