# Trait agreement (`hires.analysis`)

!!! warning "Legacy module"

    `hires.analysis` predates the 2.0 API and is scheduled for removal. Its
    functions are kept working but do not follow the conventions of the rest
    of the package, and their signatures may change in a minor release.

Plots that compare the traits of matched ground-truth and predicted objects:
distributions, bias, Bland-Altman, Taylor and target diagrams. The module
needs the `analysis` extra:

```bash
pip install "hireseg[analysis]"
```

Every function takes the table returned by
[`MatchResult.to_frame`][hires.metrics.MatchResult.to_frame]; see
[Evaluation](../user-guide/evaluation.md#trait-agreement).

::: hires.analysis
    options:
      show_root_heading: false
      show_root_toc_entry: false
      show_source: false
