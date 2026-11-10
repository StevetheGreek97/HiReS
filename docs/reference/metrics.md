# Evaluation (`hires.metrics`)

Comparing predicted annotations with ground truth. Every function takes the
**ground truth first** and the **predictions second**, as two
[`Collection`][hires.Collection] objects (one image) or two
[`Album`][hires.Album] objects (a dataset, paired by collection name). See
[Evaluation](../user-guide/evaluation.md) for the protocol and the
definition of every metric.

| Object | Description |
|--------|-------------|
| [`hires.evaluate`][hires.evaluate] | Write a complete report to a directory |
| [`match`][hires.metrics.match] | Pair predictions with ground truth by IoU |
| [`MatchResult`][hires.metrics.MatchResult], [`Match`][hires.metrics.Match] | The result of matching |
| [`detection_report`][hires.metrics.detection_report] | Precision, recall, F1 and AP per class |
| [`confusion_matrix`][hires.metrics.confusion_matrix] | Ground-truth versus predicted class counts |
| [`precision_recall_curve`][hires.metrics.precision_recall_curve] | Precision and recall per confidence threshold |
| [`average_precision`][hires.metrics.average_precision] | AP per class at one IoU threshold |
| [`iou`][hires.metrics.iou], [`iou_matrix`][hires.metrics.iou_matrix] | Polygon intersection-over-union |

## One-call report

::: hires.evaluate

## Matching

::: hires.metrics.match

::: hires.metrics.MatchResult

::: hires.metrics.Match

::: hires.metrics.STATUSES

## Metrics

::: hires.metrics.detection_report

::: hires.metrics.confusion_matrix

::: hires.metrics.precision_recall_curve

::: hires.metrics.average_precision

## Intersection over union

::: hires.metrics.iou

::: hires.metrics.iou_matrix
