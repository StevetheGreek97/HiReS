# Changelog

All notable changes to HiReS are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0]

A rewrite of the Python API around a small, immutable data model. The
command line is largely unchanged. Code written against 1.x needs to be
updated; see the
[migration guide](https://stevethegreek97.github.io/HiReS/getting-started/migration.html).

### Added

- `hires.metrics`: IoU matching (`match`, `MatchResult`, `Match`), per-class
  `detection_report` with AP50 and AP50-95, `confusion_matrix`,
  `precision_recall_curve` and `average_precision`.
- `hires.evaluate` and the `hires evaluate` command, which write a complete
  evaluation report to a directory.
- `hires.plot`: `overlay`, `comparison`, `individual`, `annotation`,
  `confusion_matrix`, `precision_recall` and `training_curves`.
- `hires.tiling`: `tile_grid`, `iter_tiles`, `tile_image`,
  `tile_annotations`, `merge_tiles` and `parse_tile_name`.
- `hires.segment`, `hires.Segmenter` and `hires.run_pipeline`. Any object
  with a compatible `predict()` method can be used as the model.
- `hires.ClassMapping` and `hires.build_class_mapping` for translating
  class ids between labelling schemas, including classes that were split.
- `Album.balance` for reducing class imbalance in tiled training sets.
- An exception hierarchy (`HiresError`, `AnnotationFormatError`,
  `MissingDependencyError`) and `HiresWarning`.
- `hires.show_versions` for bug reports.
- `hires run --batch-size` and `hires --version`.
- Type hints throughout and a `py.typed` marker.
- A test suite, and a documentation site with a user guide and a generated
  API reference.

### Changed

- **Breaking:** everything common is imported from the top-level `hires`
  namespace. The `Settings` dataclass and the `SegmentationPipeline` and
  `PlottingPipeline` classes are replaced by functions with keyword
  arguments.
- **Breaking:** `Annotation`, `Collection` and `Album` are immutable.
  Methods that used to modify an object now return a new one.
- **Breaking:** traits are computed on demand. The image size and physical
  scale belong to the `Collection`, not to each `Annotation`.
- **Breaking:** `ultralytics` is an optional dependency (`hireseg[yolo]`).
  `scipy`, `plotnine` and `SkillMetrics` moved to `hireseg[analysis]`.
- **Breaking:** the layout of the `{stem}_debug/` directory.
- Progress is reported through the standard `logging` module instead of
  `tqdm` and `print`.
- The `hires` command returns meaningful exit codes.

### Removed

- **Breaking:** the subpackages `hires.models`, `hires.pipeline`,
  `hires.processing`, `hires.operations` and `hires.viz`.
- The `tqdm` dependency.

### Deprecated

- `hires.analysis` is legacy code and will be removed or replaced in a
  future release.

## 1.x

Releases before 2.0.0 were not tracked in this file; see the
[commit history](https://github.com/StevetheGreek97/HiReS/commits/main).

[Unreleased]: https://github.com/StevetheGreek97/HiReS/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/StevetheGreek97/HiReS/releases/tag/v2.0.0
