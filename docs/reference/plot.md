# Plotting (`hires.plot`)

Visualisation of annotations, matches and evaluation results. Image
functions return an RGB `numpy.ndarray`; chart functions return the
Matplotlib `Axes` or `Figure` they drew on and never call `plt.show()`.
Every function accepts `save=`. See
[Visualisation](../user-guide/plotting.md) for examples.

| Function | Draws | Returns |
|----------|-------|---------|
| [`overlay`][hires.plot.overlay] | Annotations on their image | RGB array |
| [`comparison`][hires.plot.comparison] | Match outcomes on the image | RGB array |
| [`individual`][hires.plot.individual] | One object: ground truth versus prediction | `Figure` |
| [`annotation`][hires.plot.annotation] | The outline of one annotation | `Axes` |
| [`confusion_matrix`][hires.plot.confusion_matrix] | A confusion matrix heat map | `Axes` |
| [`precision_recall`][hires.plot.precision_recall] | Precision-recall curves | `Axes` |
| [`training_curves`][hires.plot.training_curves] | Losses from an Ultralytics `results.csv` | `Figure` |

## Images

::: hires.plot.overlay

::: hires.plot.comparison

## Single objects

::: hires.plot.individual

::: hires.plot.annotation

## Charts

::: hires.plot.confusion_matrix

::: hires.plot.precision_recall

::: hires.plot.training_curves

## Colours

::: hires.plot.PALETTE

::: hires.plot.STATUS_COLORS
