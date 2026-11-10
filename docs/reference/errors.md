# Exceptions and warnings

All exceptions raised deliberately by HiReS derive from
[`HiresError`][hires.HiresError]; all warnings derive from
[`HiresWarning`][hires.HiresWarning]. See
[Errors, warnings and logging](../user-guide/diagnostics.md) for how to
handle them.

```text
Exception
└── HiresError
    ├── AnnotationFormatError   (also a ValueError)
    └── MissingDependencyError  (also an ImportError)

UserWarning
└── HiresWarning
```

::: hires.HiresError

::: hires.AnnotationFormatError

::: hires.MissingDependencyError

::: hires.HiresWarning
