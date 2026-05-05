from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Status values as produced by CollectionMatchMaker / PairInspector.
# In CollectionMatchMaker, left=GT and right=pred, so:
#   "fp" = unmatched left (GT annotation with no pred match)
#   "fn" = unmatched right (pred annotation with no GT match)
STATUS_MATCHED = "tp"          # matched pair (same class)
STATUS_MISSED_GT = "fp"        # GT annotation unmatched by any prediction
STATUS_EXTRA_PRED = "fn"       # prediction unmatched by any GT annotation

_IGNORED_DESCRIPTOR_SUFFIXES = {
    "index",
    "class",
    "conf",
    "collection_name",
    "crop_path",
}

def _require_columns(data: Any, columns: Sequence[str], *, context: str) -> None:
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"{context} is missing required columns: {missing_text}.")


def _require_plotnine() -> Any:
    try:
        from plotnine import (
            aes,
            coord_equal,
            element_blank,
            element_text,
            facet_wrap,
            geom_abline,
            geom_blank,
            geom_boxplot,
            geom_density,
            geom_hline,
            geom_histogram,
            geom_point,
            geom_segment,
            geom_text,
            geom_step,
            geom_violin,
            ggplot,
            labs,
            scale_color_manual,
            scale_fill_manual,
            scale_linetype_manual,
            scale_y_continuous,
            theme,
            theme_bw,
        )
    except Exception as exc:
        raise RuntimeError(
            "plotnine is required for this descriptor plot."
        ) from exc

    return SimpleNamespace(
        aes=aes,
        coord_equal=coord_equal,
        element_blank=element_blank,
        element_text=element_text,
        facet_wrap=facet_wrap,
        geom_abline=geom_abline,
        geom_blank=geom_blank,
        geom_boxplot=geom_boxplot,
        geom_density=geom_density,
        geom_hline=geom_hline,
        geom_histogram=geom_histogram,
        geom_point=geom_point,
        geom_segment=geom_segment,
        geom_text=geom_text,
        geom_step=geom_step,
        geom_violin=geom_violin,
        ggplot=ggplot,
        labs=labs,
        scale_color_manual=scale_color_manual,
        scale_fill_manual=scale_fill_manual,
        scale_linetype_manual=scale_linetype_manual,
        scale_y_continuous=scale_y_continuous,
        theme=theme,
        theme_bw=theme_bw,
    )


def _optional_scipy_stats() -> tuple[Any | None, Any | None, Any | None]:
    try:
        from scipy.stats import ks_2samp, linregress, wasserstein_distance
    except Exception:
        return None, None, None
    return ks_2samp, linregress, wasserstein_distance


def _require_skill_metrics() -> Any:
    try:
        import skill_metrics as sm
    except Exception as exc:
        raise ImportError(
            "SkillMetrics is required for this plot. Install it with "
            "'pip install SkillMetrics'."
        ) from exc
    return sm


def _normalized_class_names(
    class_names: Mapping[int | str, str] | None,
) -> dict[int, str]:
    if class_names is None:
        return {}

    normalized: dict[int, str] = {}
    for key, value in class_names.items():
        try:
            normalized[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _descriptor_columns_in_order(data: Any) -> list[str]:
    descriptor_columns: list[str] = []
    seen: set[str] = set()

    for column in data.columns:
        if not str(column).startswith("pred_"):
            continue
        suffix = str(column)[5:]
        if suffix in _IGNORED_DESCRIPTOR_SUFFIXES:
            continue
        if f"gt_{suffix}" not in data.columns:
            continue
        if suffix in seen:
            continue
        seen.add(suffix)
        descriptor_columns.append(suffix)

    return descriptor_columns


def _selected_columns(data: Any, columns: Sequence[str] | None = None) -> list[str]:
    available = available_descriptor_columns(data)
    if columns is None:
        if not available:
            raise ValueError(
                "No descriptor columns were found. Pass a descriptor-enriched "
                "table from result.pair_descriptor_table()."
            )
        return available

    selected = [str(column) for column in columns]
    if not selected:
        raise ValueError("columns cannot be empty.")

    missing = sorted(set(selected).difference(available))
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(
            "descriptor table does not contain matching pred_/gt_ columns for: "
            f"{missing_text}."
        )
    return selected


def _resolve_save_path(
    save: str | Path | None,
    default_filename: str,
) -> Path | None:
    if save is None:
        return None

    destination = Path(save)
    if destination.suffix:
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    return destination / default_filename


def _save_ggplot_if_requested(
    plot: Any | None,
    save: str | Path | None,
    default_filename: str,
) -> Path | None:
    if plot is None:
        return None

    output_path = _resolve_save_path(save, default_filename)
    if output_path is None:
        return None

    plot.save(filename=str(output_path), dpi=300, verbose=False, limitsize=False)
    return output_path


def _save_matplotlib_if_requested(
    figure: Any | None,
    save: str | Path | None,
    default_filename: str,
) -> Path | None:
    if figure is None:
        return None

    output_path = _resolve_save_path(save, default_filename)
    if output_path is None:
        return None

    figure.savefig(str(output_path), dpi=300, bbox_inches="tight")
    return output_path


def _show_or_close_matplotlib(figure: Any, show: bool) -> None:

    if show:
        try:
            from IPython.display import display

            display(figure)
        except Exception:
            plt.show()
    else:
        plt.close(figure)


def _show_plotnine(plot: Any, show: bool) -> None:
    if not show:
        return

    try:
        from io import BytesIO
        from IPython.display import Image, display

        buf = BytesIO()
        plot.save(buf, "png", verbose=False, limitsize=False)
        buf.seek(0)
        display(Image(buf.read()))
    except Exception:
        print(plot)


def _sample_column_name(data: Any) -> str | None:
    if "sample" in data.columns:
        return "sample"
    if "file_name" in data.columns:
        return "file_name"
    return None


def _ensure_sample_column(data: Any) -> Any:

    frame = data.copy()
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    sample_column = _sample_column_name(frame)
    if sample_column is None:
        frame["sample"] = "comparison"
        return frame

    frame["sample"] = frame[sample_column].astype(str)
    return frame


def _normalize_align(align: str | None) -> str | None:
    if align is None:
        return None

    key = str(align).strip().lower().replace("'", "").replace('"', "")
    if key in {"", "none", "null"}:
        return None
    if key in {"median", "median_iqr", "iqr", "med"}:
        return "median_iqr"
    if key in {"mean", "mean_std", "std", "avg", "average"}:
        return "mean_std"
    raise ValueError(
        "align must be one of {'median', 'mean', 'median_iqr', 'mean_std'}."
    )


def _filter_samples(
    data: Any,
    *,
    samples: Sequence[str | int] | None = None,
    sample_n: int | None = None,
    random_state: int | None = None,
):
    if data.empty or "sample" not in data.columns:
        return data

    out = data.copy()
    sample_series = out["sample"].astype(str)

    if samples is not None:
        keep = [str(sample) for sample in samples]
        if not keep:
            return out.iloc[0:0].copy()
        keep_set = set(keep)
        out = out[sample_series.isin(keep_set)].copy()
        sample_series = out["sample"].astype(str)

    if sample_n is not None:
        n_samples = int(sample_n)
        if n_samples <= 0:
            raise ValueError("sample_n must be a positive integer.")
        unique_samples = out["sample"].dropna().astype(str).drop_duplicates()
        if unique_samples.empty:
            return out.iloc[0:0].copy()
        if n_samples < len(unique_samples):
            chosen = unique_samples.sample(n=n_samples, random_state=random_state).tolist()
            out = out[sample_series.isin(set(chosen))].copy()

    return out


def _resolve_excluded_class_ids(
    exclude_classes: Sequence[int | str] | None,
    *,
    class_names: Mapping[int | str, str] | None = None,
) -> set[int]:
    if exclude_classes is None:
        return set()

    normalized_names = _normalized_class_names(class_names)
    reverse_name_map = {
        str(name).strip().lower(): class_id
        for class_id, name in normalized_names.items()
    }

    excluded_ids: set[int] = set()
    for item in exclude_classes:
        if isinstance(item, (int, np.integer)):
            excluded_ids.add(int(item))
            continue

        token = str(item).strip()
        if not token:
            continue

        try:
            excluded_ids.add(int(token))
            continue
        except ValueError:
            pass

        class_id = reverse_name_map.get(token.lower())
        if class_id is not None:
            excluded_ids.add(int(class_id))
            continue

        raise ValueError(f"Unknown class specifier in exclude_classes: {item!r}.")

    return excluded_ids


def _class_items_for_frame(
    data: Any,
    *,
    class_names: Mapping[int | str, str] | None = None,
    exclude_classes: Sequence[int | str] | None = None,
) -> list[tuple[int, str]]:
    normalized_names = _normalized_class_names(class_names)
    excluded_ids = _resolve_excluded_class_ids(
        exclude_classes,
        class_names=class_names,
    )

    class_ids: set[int] = set()
    for column in ("gt_class", "pred_class"):
        if column not in data.columns:
            continue
        values = data[column].dropna().unique().tolist()
        class_ids.update(int(value) for value in values)

    ordered_ids = sorted(class_id for class_id in class_ids if class_id not in excluded_ids)
    return [
        (class_id, normalized_names.get(class_id, str(class_id)))
        for class_id in ordered_ids
    ]


def _align_values(source: np.ndarray, target: np.ndarray, *, method: str) -> np.ndarray:
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    source_array = source_array[np.isfinite(source_array)]
    target_array = target_array[np.isfinite(target_array)]

    if source_array.size == 0:
        return source_array
    if target_array.size == 0:
        return source_array.copy()

    if method == "median_iqr":
        source_center = float(np.nanmedian(source_array))
        target_center = float(np.nanmedian(target_array))
        source_q1, source_q3 = np.nanpercentile(source_array, [25, 75])
        target_q1, target_q3 = np.nanpercentile(target_array, [25, 75])
        source_scale = float(source_q3 - source_q1)
        target_scale = float(target_q3 - target_q1)
    elif method == "mean_std":
        source_center = float(np.nanmean(source_array))
        target_center = float(np.nanmean(target_array))
        source_scale = float(np.nanstd(source_array, ddof=1)) if source_array.size > 1 else 0.0
        target_scale = float(np.nanstd(target_array, ddof=1)) if target_array.size > 1 else 0.0
    else:
        raise ValueError(f"Unknown alignment method: {method!r}.")

    if np.isfinite(source_scale) and np.isfinite(target_scale) and source_scale > 0 and target_scale > 0:
        return (source_array - source_center) * (target_scale / source_scale) + target_center

    return source_array - source_center + target_center


def _processed_descriptor_frame(
    data: Any,
    *,
    columns: Sequence[str] | None = None,
    only_tp: bool = True,
    log: bool = False,
    align: str | None = "median",
    class_names: Mapping[int | str, str] | None = None,
    exclude_classes: Sequence[int | str] | None = None,
    file_names: Sequence[str] | None = None,
    samples: Sequence[str | int] | None = None,
    sample_n: int | None = None,
    random_state: int | None = None,
) -> tuple[Any, list[str], list[tuple[int, str]]]:

    # auto-convert raw PairInspector.pairs_df() output (left_*/right_* columns)
    if isinstance(data, pd.DataFrame) and "left_class" in data.columns and "gt_class" not in data.columns:
        data = from_pairs_df(data)

    _require_columns(
        data,
        ("status", "class_match", "pred_class", "gt_class"),
        context="descriptor table",
    )

    frame = _ensure_sample_column(data)

    if file_names is not None:
        if "file_name" not in frame.columns:
            raise KeyError(
                "file_names filtering requires a path-level descriptor table "
                "with a 'file_name' column."
            )
        allowed_file_names = {str(file_name) for file_name in file_names}
        frame = frame[frame["file_name"].astype(str).isin(allowed_file_names)].copy()

    frame = _filter_samples(
        frame,
        samples=samples,
        sample_n=sample_n,
        random_state=random_state,
    )

    selected_columns = _selected_columns(frame, columns)
    class_items = _class_items_for_frame(
        frame,
        class_names=class_names,
        exclude_classes=exclude_classes,
    )

    normalized_names = _normalized_class_names(class_names)
    frame["gt_species"] = frame["gt_class"].map(
        lambda value: (
            normalized_names.get(int(value), str(int(value)))
            if pd.notna(value)
            else np.nan
        )
    )
    frame["pred_species"] = frame["pred_class"].map(
        lambda value: (
            normalized_names.get(int(value), str(int(value)))
            if pd.notna(value)
            else np.nan
        )
    )

    align_key = _normalize_align(align)
    if align_key is not None:
        for class_id, _ in class_items:
            for column in selected_columns:
                gt_column = f"gt_{column}"
                pred_column = f"pred_{column}"
                mask = (
                    frame["status"].eq(STATUS_MATCHED)
                    & frame["class_match"].eq(True)
                    & frame["gt_class"].eq(float(class_id))
                    & frame[gt_column].notna()
                    & frame[pred_column].notna()
                )
                if int(mask.sum()) < 2:
                    continue
                aligned = _align_values(
                    frame.loc[mask, pred_column].to_numpy(dtype=float),
                    frame.loc[mask, gt_column].to_numpy(dtype=float),
                    method=align_key,
                )
                frame.loc[mask, pred_column] = aligned

    if log:
        for column in selected_columns:
            gt_column = f"gt_{column}"
            pred_column = f"pred_{column}"
            gt_mask = frame[gt_column] > 0
            pred_mask = frame[pred_column] > 0
            frame.loc[gt_mask, gt_column] = np.log10(frame.loc[gt_mask, gt_column].to_numpy(dtype=float))
            frame.loc[~gt_mask, gt_column] = np.nan
            frame.loc[pred_mask, pred_column] = np.log10(frame.loc[pred_mask, pred_column].to_numpy(dtype=float))
            frame.loc[~pred_mask, pred_column] = np.nan

    if only_tp:
        frame = frame[
            frame["status"].eq(STATUS_MATCHED) & frame["class_match"].eq(True)
        ].copy()

    return frame, selected_columns, class_items


def _histogram_bins(
    values: np.ndarray,
    bins: int | Sequence[float],
) -> int | np.ndarray:
    if not isinstance(bins, int):
        return np.asarray(list(bins), dtype=float)

    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return bins

    value_min = float(np.min(finite_values))
    value_max = float(np.max(finite_values))
    if not np.isclose(value_min, value_max):
        return bins

    pad = 0.5 if value_min == 0.0 else abs(value_min) * 0.05
    if pad == 0.0:
        pad = 0.5
    return np.linspace(value_min - pad, value_max + pad, int(bins) + 1)


def _ecdf_xy(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    x = np.sort(array)
    y = np.arange(1, array.size + 1, dtype=float) / float(array.size)
    return x, y


def from_pairs_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a :class:`PairInspector.pairs_df` result into the format expected by this module.

    ``PairInspector.pairs_df`` uses ``left_*`` (GT) / ``right_*`` (pred) column prefixes
    and status values ``"tp" | "misclassified" | "fp" | "fn"``.  This function produces
    the ``gt_*`` / ``pred_*`` naming convention and adds the ``class_match`` column
    required by all plotting functions in this module.

    The ``"misclassified"`` status (matched pair, wrong class) is normalised to
    ``STATUS_MATCHED`` (``"tp"``) with ``class_match=False`` so the matched-pair
    descriptors are available for distributions / Bland-Altman while the class-match
    filter still excludes them from TP-only analyses.
    """
    out = df.copy()

    # rename left_* → gt_*, right_* → pred_*
    renames: dict[str, str] = {}
    for col in out.columns:
        if col.startswith("left_"):
            renames[col] = "gt_" + col[5:]
        elif col.startswith("right_"):
            renames[col] = "pred_" + col[6:]
    out = out.rename(columns=renames)

    # normalise "misclassified" → STATUS_MATCHED so matched descriptors are reachable
    if "status" in out.columns:
        out["status"] = out["status"].replace("misclassified", STATUS_MATCHED)

    # class_match: True only when both classes are present AND equal
    if "gt_class" in out.columns and "pred_class" in out.columns:
        both_present = out["gt_class"].notna() & out["pred_class"].notna()
        same_class = out["gt_class"].eq(out["pred_class"])
        out["class_match"] = both_present & same_class
    else:
        out["class_match"] = False

    # add index stubs for descriptor_distribution_table compatibility
    if "gt_index" not in out.columns:
        out["gt_index"] = pd.NA
    if "pred_index" not in out.columns:
        out["pred_index"] = pd.NA

    # expose collection_name as file_name for per-file filtering
    if "collection_name" in out.columns and "file_name" not in out.columns:
        out = out.rename(columns={"collection_name": "file_name"})

    return out


def available_descriptor_columns(data: Any) -> list[str]:
    """
    Return descriptor names present on both the prediction and GT sides.

    This inspects a descriptor-enriched pair table such as the output of
    ``result.pair_descriptor_table()`` and returns suffixes like
    ``["area", "perimeter", "solidity"]``.
    """
    return _descriptor_columns_in_order(data)


def filter_descriptor_rows(
    data: Any,
    *,
    only_tp: bool = True,
    statuses: Sequence[str] | None = None,
    class_match: bool | None = None,
    file_names: Sequence[str] | None = None,
):
    """
    Filter a descriptor table before long-form expansion.

    ``only_tp=True`` keeps only matched rows with ``class_match == True``.
    """

    _require_columns(
        data,
        ("status", "class_match", "pred_class", "gt_class"),
        context="descriptor table",
    )

    filtered = data.copy()
    if not isinstance(filtered, pd.DataFrame):
        filtered = pd.DataFrame(filtered)

    if only_tp:
        filtered = filtered[
            filtered["status"].eq(STATUS_MATCHED) & filtered["class_match"].eq(True)
        ].copy()

    if statuses is not None:
        allowed_statuses = {str(status) for status in statuses}
        filtered = filtered[filtered["status"].isin(allowed_statuses)].copy()

    if class_match is not None:
        filtered = filtered[filtered["class_match"].eq(bool(class_match))].copy()

    if file_names is not None:
        if "file_name" not in filtered.columns:
            raise KeyError(
                "file_names filtering requires a path-level descriptor table "
                "with a 'file_name' column."
            )
        allowed_file_names = {str(file_name) for file_name in file_names}
        filtered = filtered[filtered["file_name"].astype(str).isin(allowed_file_names)].copy()

    return filtered


def descriptor_distribution_table(
    data: Any,
    *,
    columns: Sequence[str] | None = None,
    only_tp: bool = True,
    statuses: Sequence[str] | None = None,
    class_match: bool | None = None,
    class_ids: Sequence[int] | None = None,
    class_names: Mapping[int | str, str] | None = None,
    exclude_classes: Sequence[int | str] | None = None,
    file_names: Sequence[str] | None = None,
    include_gt: bool = True,
    include_pred: bool = True,
    log: bool = False,
):
    """
    Return a long-form descriptor table suitable for plotting.

    The output contains one row per descriptor value with columns:
    ``descriptor``, ``source``, ``class_id``, ``class_name``, ``value``,
    plus the original pair status metadata.
    """
    if not include_gt and not include_pred:
        raise ValueError("At least one of include_gt or include_pred must be True.")

    filtered = filter_descriptor_rows(
        data,
        only_tp=only_tp,
        statuses=statuses,
        class_match=class_match,
        file_names=file_names,
    )

    selected_columns = _selected_columns(filtered, columns)

    base_columns = ["status", "class_match"]
    if "file_name" in filtered.columns:
        base_columns.append("file_name")
    if "sample" in filtered.columns:
        base_columns.append("sample")

    parts: list[Any] = []
    for descriptor in selected_columns:
        if include_gt:
            gt_columns = list(base_columns) + ["gt_index", "gt_class", f"gt_{descriptor}"]
            _require_columns(filtered, gt_columns[2:], context="descriptor table")
            gt_part = filtered.loc[:, gt_columns].copy()
            gt_part = gt_part.rename(
                columns={
                    "gt_index": "annotation_index",
                    "gt_class": "class_id",
                    f"gt_{descriptor}": "value",
                }
            )
            gt_part["source"] = "gt"
            gt_part["descriptor"] = descriptor
            parts.append(gt_part)

        if include_pred:
            pred_columns = list(base_columns) + ["pred_index", "pred_class", f"pred_{descriptor}"]
            _require_columns(filtered, pred_columns[2:], context="descriptor table")
            pred_part = filtered.loc[:, pred_columns].copy()
            pred_part = pred_part.rename(
                columns={
                    "pred_index": "annotation_index",
                    "pred_class": "class_id",
                    f"pred_{descriptor}": "value",
                }
            )
            pred_part["source"] = "pred"
            pred_part["descriptor"] = descriptor
            parts.append(pred_part)

    if not parts:
        return pd.DataFrame(
            columns=[
                "descriptor",
                "source",
                "class_id",
                "class_name",
                "annotation_index",
                "value",
                "status",
                "class_match",
            ]
        )

    long_table = pd.concat(parts, ignore_index=True)
    long_table = long_table[long_table["class_id"].notna() & long_table["value"].notna()].copy()
    if long_table.empty:
        long_table["class_name"] = pd.Series(dtype=object)
        return long_table

    long_table["class_id"] = long_table["class_id"].astype(int)
    long_table["annotation_index"] = long_table["annotation_index"].astype("Int64")
    long_table["value"] = long_table["value"].astype(float)

    excluded_ids = _resolve_excluded_class_ids(
        exclude_classes,
        class_names=class_names,
    )
    if excluded_ids:
        long_table = long_table[~long_table["class_id"].isin(excluded_ids)].copy()

    if class_ids is not None:
        allowed_class_ids = {int(class_id) for class_id in class_ids}
        long_table = long_table[long_table["class_id"].isin(allowed_class_ids)].copy()

    if log:
        long_table = long_table[long_table["value"] > 0].copy()
        long_table["value"] = np.log10(long_table["value"].to_numpy(dtype=float))

    normalized_names = _normalized_class_names(class_names)
    long_table["class_name"] = long_table["class_id"].map(
        lambda class_id: normalized_names.get(int(class_id), str(int(class_id)))
    )

    source_dtype = pd.CategoricalDtype(categories=["gt", "pred"], ordered=True)
    long_table["source"] = long_table["source"].astype(source_dtype)

    descriptor_dtype = pd.CategoricalDtype(
        categories=selected_columns,
        ordered=True,
    )
    long_table["descriptor"] = long_table["descriptor"].astype(descriptor_dtype)

    class_order = sorted(long_table["class_id"].unique().tolist())
    class_name_order = [
        normalized_names.get(int(class_id), str(int(class_id)))
        for class_id in class_order
    ]
    class_dtype = pd.CategoricalDtype(
        categories=class_name_order,
        ordered=True,
    )
    long_table["class_name"] = long_table["class_name"].astype(class_dtype)

    sort_columns = ["class_name", "descriptor", "source"]
    if "file_name" in long_table.columns:
        sort_columns.insert(0, "file_name")
    if "sample" in long_table.columns:
        sort_columns.insert(0, "sample")
    return long_table.sort_values(sort_columns).reset_index(drop=True)


def distributions(
    data: Any,
    *,
    show: bool = True,
    only_tp: bool = True,
    columns: Sequence[str] | None = None,
    log: bool = False,
    align: str | None = "median",
    save: str | Path | None = None,
    samples: Sequence[str | int] | None = None,
    sample_n: int | None = None,
    random_state: int | None = None,
    class_names: Mapping[int | str, str] | None = None,
    exclude_classes: Sequence[int | str] | None = None,
    file_names: Sequence[str] | None = None,
    bins: int = 40,
    kde: bool = True,
) -> tuple[Any, Any | None]:
    """
    Plot GT vs Pred descriptor distributions using plotnine (ggplot).

    This is the function-based equivalent of the older
    ``DescriptorComarison.distributions(...)`` implementation.
    It returns the processed wide descriptor table plus the ggplot object.
    """
    plotnine = _require_plotnine()
   

    processed, selected_columns, class_items = _processed_descriptor_frame(
        data,
        columns=columns,
        only_tp=only_tp,
        log=log,
        align=align,
        class_names=class_names,
        exclude_classes=exclude_classes,
        file_names=file_names,
        samples=samples,
        sample_n=sample_n,
        random_state=random_state,
    )
    if processed.empty or not class_items:
        return processed, None

    pred_set_label = "Pred (TP only)" if only_tp else "Pred"
    species_order = [str(name) for _, name in class_items]
    descriptor_order = [str(column) for column in selected_columns]
    panel_order = [
        f"{species} | {descriptor}"
        for species in species_order
        for descriptor in descriptor_order
    ]

    long_rows: list[dict[str, Any]] = []
    ann_rows: list[dict[str, Any]] = []

    for class_id, species in class_items:
        species_label = str(species)
        for column in descriptor_order:
            panel = f"{species_label} | {column}"
            gt_column = f"gt_{column}"
            pred_column = f"pred_{column}"

            if only_tp:
                subset = processed[
                    processed["status"].eq(STATUS_MATCHED)
                    & processed["gt_class"].eq(float(class_id))
                    & processed["pred_class"].eq(float(class_id))
                ]
                gt_vals = subset[gt_column].to_numpy(dtype=float)
                pred_vals = subset[pred_column].to_numpy(dtype=float)
            else:
                gt_vals = processed[
                    processed["status"].isin([STATUS_MATCHED, STATUS_MISSED_GT])
                    & processed["gt_class"].eq(float(class_id))
                ][gt_column].to_numpy(dtype=float)
                pred_vals = processed[
                    processed["status"].isin([STATUS_MATCHED, STATUS_EXTRA_PRED])
                    & processed["pred_class"].eq(float(class_id))
                ][pred_column].to_numpy(dtype=float)

            gt_vals = gt_vals[np.isfinite(gt_vals)]
            pred_vals = pred_vals[np.isfinite(pred_vals)]

            if gt_vals.size:
                long_rows.extend(
                    {
                        "panel": panel,
                        "species": species_label,
                        "descriptor": str(column),
                        "set": "GT",
                        "value": float(value),
                    }
                    for value in gt_vals
                )
            if pred_vals.size:
                long_rows.extend(
                    {
                        "panel": panel,
                        "species": species_label,
                        "descriptor": str(column),
                        "set": pred_set_label,
                        "value": float(value),
                    }
                    for value in pred_vals
                )

            if gt_vals.size and pred_vals.size:
                both = np.concatenate([gt_vals, pred_vals])
                x_min = float(np.min(both))
                x_max = float(np.max(both))
                if x_min == x_max:
                    x_min -= 0.5
                    x_max += 0.5

                hist_bins = _histogram_bins(both, int(bins))
                hgt, _ = np.histogram(gt_vals, bins=hist_bins, density=True)
                hpr, _ = np.histogram(pred_vals, bins=hist_bins, density=True)
                ymax = float(np.nanmax(np.concatenate([hgt, hpr])))
                if not np.isfinite(ymax) or ymax <= 0:
                    ymax = 1.0

                x_text = x_min + 0.02 * (x_max - x_min if x_max > x_min else 1.0)
                y_text = ymax * 0.98
                mean_diff = float(np.mean(pred_vals) - np.mean(gt_vals))
                ann_rows.append(
                    {
                        "species": species_label,
                        "descriptor": str(column),
                        "panel": panel,
                        "x_lab": x_text,
                        "y_lab": y_text,
                        "label": (
                            f"nGT={int(gt_vals.size)}, nPred={int(pred_vals.size)}\n"
                            f"Δmean={mean_diff:.2g}"
                        ),
                    }
                )

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return processed, None

    observed_panels = [
        panel for panel in panel_order if panel in set(long_df["panel"].astype(str).tolist())
    ]
    if not observed_panels:
        return processed, None

    long_df["species"] = pd.Categorical(long_df["species"], categories=species_order, ordered=True)
    long_df["descriptor"] = pd.Categorical(long_df["descriptor"], categories=descriptor_order, ordered=True)
    long_df["panel"] = pd.Categorical(long_df["panel"], categories=observed_panels, ordered=True)
    long_df["set"] = pd.Categorical(long_df["set"], categories=["GT", pred_set_label], ordered=True)

    if ann_rows:
        ann_df = pd.DataFrame(ann_rows)
        ann_df["species"] = pd.Categorical(ann_df["species"], categories=species_order, ordered=True)
        ann_df["descriptor"] = pd.Categorical(ann_df["descriptor"], categories=descriptor_order, ordered=True)
        ann_df = ann_df[ann_df["panel"].isin(observed_panels)].copy()
        ann_df["panel"] = pd.Categorical(ann_df["panel"], categories=observed_panels, ordered=True)
    else:
        ann_df = pd.DataFrame(columns=["species", "descriptor", "x_lab", "y_lab", "label", "panel"])

    nrows = len(species_order)
    ncols = len(descriptor_order)
    width = max(4.0 * ncols, 8.0)
    height = max(2.8 * nrows, 4.0)

    plot = (
        plotnine.ggplot(long_df, plotnine.aes(x="value", fill="set", color="set"))
        + plotnine.geom_histogram(
            plotnine.aes(y="..density.."),
            bins=int(bins),
            alpha=0.25,
            position="identity",
            size=0.25,
        )
        + plotnine.facet_wrap("~panel", ncol=ncols, scales="free")
        + plotnine.scale_fill_manual(values={"GT": "#1f77b4", pred_set_label: "#d62728"})
        + plotnine.scale_color_manual(values={"GT": "#1f77b4", pred_set_label: "#d62728"})
        + plotnine.labs(
            x="log10(value)" if log else "value",
            y="Density",
            title=f"GT vs {pred_set_label} Descriptor Distributions",
        )
        + plotnine.theme_bw()
        + plotnine.theme(
            figure_size=(width, height),
            axis_text_x=plotnine.element_text(size=8),
            axis_text_y=plotnine.element_text(size=8),
            strip_text=plotnine.element_text(weight="bold"),
            legend_title=plotnine.element_text(size=9),
            legend_text=plotnine.element_text(size=8),
            subplots_adjust={"wspace": 0.12, "hspace": 0.12},
        )
    )

    if kde:
        kde_ok = False
        grouped = long_df.groupby(["panel", "set"], observed=False)["value"]
        for _, vals in grouped:
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size >= 2 and float(np.min(arr)) < float(np.max(arr)):
                kde_ok = True
                break
        if kde_ok:
            plot = plot + plotnine.geom_density(alpha=0.0, size=1.0)

    if not ann_df.empty:
        plot = plot + plotnine.geom_text(
            data=ann_df,
            mapping=plotnine.aes(x="x_lab", y="y_lab", label="label"),
            inherit_aes=False,
            ha="left",
            va="top",
            size=7,
        )

    _save_ggplot_if_requested(plot, save, "distributions.png")
    _show_plotnine(plot, show)
    return processed, plot



def bias(
    data: Any,
    *,
    show: bool = True,
    log: bool = True,
    align: str | None = "median",
    columns: Sequence[str] | None = None,
    only_tp: bool = True,
    onlytp: bool | None = None,
    save: str | Path | None = None,
    class_names: Mapping[int | str, str] | None = None,
    exclude_classes: Sequence[int | str] | None = None,
    file_names: Sequence[str] | None = None,
    samples: Sequence[str | int] | None = None,
    sample_n: int | None = None,
    random_state: int | None = None,
):
    plotnine = _require_plotnine()
  

    align_key = _normalize_align(align)
    tp_only = bool(only_tp if onlytp is None else onlytp)
    frame, selected_columns, class_items = _processed_descriptor_frame(
        data,
        columns=columns,
        only_tp=tp_only,
        log=False,
        align=align_key,
        class_names=class_names,
        exclude_classes=exclude_classes,
        file_names=file_names,
        samples=samples,
        sample_n=sample_n,
        random_state=random_state,
    )
    if not class_items:
        return pd.DataFrame(columns=["sample", "species", "descriptor", "diff_pct"])

    species_order = [str(name) for _, name in class_items]
    descriptor_order = [str(column) for column in selected_columns]
    mode_label = "TP" if tp_only else "matched-pair"

    rows: list[dict[str, Any]] = []
    for class_id, species in class_items:
        species_label = str(species)
        for column in descriptor_order:
            gt_column = f"gt_{column}"
            pred_column = f"pred_{column}"

            subset = frame[frame["gt_class"].eq(float(class_id))].copy()
            pair = subset[["sample", gt_column, pred_column]].copy()
            mask = np.isfinite(pair[gt_column]) & np.isfinite(pair[pred_column]) & (pair[gt_column] > 0)
            pair = pair.loc[mask]
            if pair.empty:
                continue

            gt_values = pair[gt_column].to_numpy(dtype=float)
            pred_values = pair[pred_column].to_numpy(dtype=float)

            if bool(log):
                diff_pct = (pred_values / gt_values - 1.0) * 100.0
            else:
                diff_pct = (pred_values / gt_values - 1.0) * 100.0

            rows.extend(
                {
                    "sample": str(sample),
                    "species": species_label,
                    "descriptor": str(column),
                    "diff_pct": float(value),
                }
                for sample, value in zip(pair["sample"].astype(str).tolist(), diff_pct.tolist())
            )

    bias_df = pd.DataFrame(rows)
    if bias_df.empty:
        return pd.DataFrame(columns=["sample", "species", "descriptor", "diff_pct"])

    bias_df["species"] = pd.Categorical(bias_df["species"], categories=species_order, ordered=True)
    bias_df["descriptor"] = pd.Categorical(
        bias_df["descriptor"],
        categories=descriptor_order,
        ordered=True,
    )
    panel_order = [f"{species} | {descriptor}" for species in species_order for descriptor in descriptor_order]
    bias_df["panel"] = pd.Categorical(
        bias_df["species"].astype(str) + " | " + bias_df["descriptor"].astype(str),
        categories=panel_order,
        ordered=True,
    )
    bias_df["xslot"] = "distribution"

    if align_key is None:
        bias_df["panel_center"] = 0.0
        bias_df["diff_plot_pct"] = bias_df["diff_pct"]
        center_name = "none"
        y_label = "Percent difference (Pred/GT - 1) * 100"
        title = f"{mode_label} bias per species and descriptor"
    else:
        center_func = "median" if align_key == "median_iqr" else "mean"
        center_name = "median" if align_key == "median_iqr" else "mean"
        centers = (
            bias_df.groupby(["species", "descriptor"], as_index=False, observed=False)["diff_pct"]
            .agg(panel_center=center_func)
        )
        bias_df = bias_df.merge(centers, on=["species", "descriptor"], how="left")
        bias_df["diff_plot_pct"] = bias_df["diff_pct"] - bias_df["panel_center"]
        y_label = f"Centered percent difference ((Pred/GT - 1)*100 - panel {center_name})"
        title = f"{mode_label} {center_name}-centered bias per species and descriptor"

    bias_df["diff_centered_pct"] = bias_df["diff_plot_pct"]

    annotations = (
        bias_df.groupby(["species", "descriptor"], as_index=False, observed=False)
        .agg(
            n=("diff_plot_pct", "size"),
            center=("panel_center", "first"),
            ymin=("diff_plot_pct", "min"),
            ymax=("diff_plot_pct", "max"),
        )
    )
    ranges = annotations["ymax"] - annotations["ymin"]
    annotations["y"] = annotations["ymin"] + 0.06 * ranges.replace(0, 1.0)
    annotations["label"] = annotations.apply(
        lambda row: (
            f"n={int(row['n'])}\n{center_name}={float(row['center']):.2f}%"
            if center_name != "none"
            else f"n={int(row['n'])}"
        ),
        axis=1,
    )
    annotations["panel"] = pd.Categorical(
        annotations["species"].astype(str) + " | " + annotations["descriptor"].astype(str),
        categories=panel_order,
        ordered=True,
    )
    annotations["xslot"] = "distribution"

    nrows = len(species_order)
    ncols = len(descriptor_order)
    plot = (
        plotnine.ggplot(bias_df, plotnine.aes(x="xslot", y="diff_plot_pct"))
        + plotnine.geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
        + plotnine.geom_violin(fill="#8ecae6", color="#1f2937", alpha=0.60, trim=True)
        + plotnine.geom_boxplot(width=0.12, fill="white", color="black", outlier_alpha=0)
        + plotnine.facet_wrap("~ panel", ncol=ncols, scales="free_y", drop=False)
        + plotnine.labs(
            x="Descriptor",
            y=y_label,
            title=title,
        )
        + plotnine.theme_bw()
        + plotnine.theme(
            figure_size=(4.1 * ncols, 2.8 * nrows),
            axis_text_x=plotnine.element_blank(),
            axis_ticks_major_x=plotnine.element_blank(),
            axis_text_y=plotnine.element_text(size=8),
            strip_text=plotnine.element_text(weight="bold"),
            subplots_adjust={"wspace": 0.12, "hspace": 0.12},
        )
    )

    y_tick_step = 8.0
    y_min = float(np.floor(bias_df["diff_plot_pct"].min() / y_tick_step) * y_tick_step)
    y_max = float(np.ceil(bias_df["diff_plot_pct"].max() / y_tick_step) * y_tick_step)
    y_breaks = list(np.arange(y_min, y_max + y_tick_step, y_tick_step))
    plot = plot + plotnine.scale_y_continuous(breaks=y_breaks)

    if not annotations.empty:
        plot = plot + plotnine.geom_text(
            data=annotations,
            mapping=plotnine.aes(x="xslot", y="y", label="label"),
            inherit_aes=False,
            ha="right",
            va="bottom",
            nudge_x=0.24,
            size=6.5,
        )

    _save_ggplot_if_requested(plot, save, "bias.png")
    _show_plotnine(plot, show)
    return bias_df


def bland_altman(
    data: Any,
    *,
    show: bool = True,
    align: str | None = "median",
    log: bool = True,
    columns: Sequence[str] | None = None,
    stats_box_loc: str = "upper right",
    panel_stats: bool = True,
    save: str | Path | None = None,
    class_names: Mapping[int | str, str] | None = None,
    exclude_classes: Sequence[int | str] | None = None,
    file_names: Sequence[str] | None = None,
    samples: Sequence[str | int] | None = None,
    sample_n: int | None = None,
    random_state: int | None = None,
) -> tuple[Any, Any | None]:
    plotnine = _require_plotnine()
 
    _, linregress, _ = _optional_scipy_stats()

    def stats_box_anchor(location: str) -> tuple[float, float, str, str]:
        lookup = {
            "upper left": (0.02, 0.98, "left", "top"),
            "upper right": (0.98, 0.98, "right", "top"),
            "lower left": (0.02, 0.02, "left", "bottom"),
            "lower right": (0.98, 0.02, "right", "bottom"),
        }
        return lookup.get(str(location).strip().lower(), lookup["upper left"])

    def auto_stats_box_loc(x_values: np.ndarray, y_values: np.ndarray) -> str:
        if x_values.size < 6:
            return "upper right"
        x_mid = float(np.median(x_values))
        y_mid = float(np.median(y_values))
        counts = {
            "upper left": int(np.sum((x_values <= x_mid) & (y_values >= y_mid))),
            "upper right": int(np.sum((x_values > x_mid) & (y_values >= y_mid))),
            "lower left": int(np.sum((x_values <= x_mid) & (y_values < y_mid))),
            "lower right": int(np.sum((x_values > x_mid) & (y_values < y_mid))),
        }
        return min(counts, key=counts.get)

    def expand_y_for_stats(y_lo: float, y_hi: float, location: str | None) -> tuple[float, float]:
        if not panel_stats or location is None:
            return y_lo, y_hi
        span = y_hi - y_lo
        if span <= 0:
            span = 1.0
        extra = 0.35 * span
        location_key = str(location).strip().lower()
        if location_key.startswith("upper"):
            return y_lo, y_hi + extra
        if location_key.startswith("lower"):
            return y_lo - extra, y_hi
        return y_lo - 0.5 * extra, y_hi + 0.5 * extra

    frame, selected_columns, class_items = _processed_descriptor_frame(
        data,
        columns=columns,
        only_tp=True,
        log=False,
        align=align,
        class_names=class_names,
        exclude_classes=exclude_classes,
        file_names=file_names,
        samples=samples,
        sample_n=sample_n,
        random_state=random_state,
    )
    frame = frame[
        frame["status"].eq(STATUS_MATCHED) & frame["class_match"].eq(True)
    ].copy()
    if frame.empty or not class_items:
        return pd.DataFrame(columns=["species", "descriptor", "x", "y"]), None

    ncols = len(selected_columns)
    descriptor_label_map = {str(column): str(column).replace("_", " ") for column in selected_columns}
    descriptor_order = [descriptor_label_map[str(column)] for column in selected_columns]
    cache: dict[tuple[int, int], dict[str, Any] | None] = {}

    for row_index, (class_id, class_name) in enumerate(class_items):
        for column_index, column in enumerate(selected_columns):
            column = str(column)
            gt_column = f"gt_{column}"
            pred_column = f"pred_{column}"

            subset = frame[
                frame["gt_class"].eq(float(class_id))
                & frame["pred_class"].eq(float(class_id))
            ].copy()

            pair = subset[[gt_column, pred_column]].copy()
            mask = np.isfinite(pair[gt_column]) & np.isfinite(pair[pred_column])
            mask = mask & (pair[gt_column] > 0) & (pair[pred_column] > 0)
            pair = pair.loc[mask]

            if len(pair) < 2:
                cache[(row_index, column_index)] = None
                continue

            gt_values = pair[gt_column].to_numpy(dtype=float)
            pred_values = pair[pred_column].to_numpy(dtype=float)
            if bool(log):
                gt_transformed = np.log10(gt_values)
                pred_transformed = np.log10(pred_values)
            else:
                gt_transformed = np.log(gt_values)
                pred_transformed = np.log(pred_values)

            diff = pred_transformed - gt_transformed
            mean_axis = 0.5 * (gt_transformed + pred_transformed)
            mean_diff = float(np.mean(diff))

            regression = None
            if linregress is not None and mean_axis.size >= 2:
                spread = float(np.max(mean_axis) - np.min(mean_axis))
                if spread > 0:
                    regression = linregress(mean_axis, diff)

            x0 = float(np.min(mean_axis))
            x1 = float(np.max(mean_axis))
            xpad = 0.08 * (x1 - x0 if x1 > x0 else 1.0)
            x_lo, x_hi = x0 - xpad, x1 + xpad

            y_candidates = [float(np.min(diff)), float(np.max(diff)), 0.0, mean_diff]
            if regression is not None:
                y_candidates.extend(
                    [
                        float(regression.intercept + regression.slope * x_lo),
                        float(regression.intercept + regression.slope * x_hi),
                    ]
                )
            y0 = float(np.min(y_candidates))
            y1 = float(np.max(y_candidates))
            ypad = 0.08 * (y1 - y0 if y1 > y0 else 1.0)
            y_lo, y_hi = y0 - ypad, y1 + ypad
            stats_loc = None
            if panel_stats:
                stats_loc = (
                    auto_stats_box_loc(mean_axis, diff)
                    if str(stats_box_loc).strip().lower() == "auto"
                    else str(stats_box_loc)
                )
                y_lo, y_hi = expand_y_for_stats(y_lo, y_hi, stats_loc)

            cache[(row_index, column_index)] = {
                "class_name": str(class_name).replace("_", " "),
                "descriptor": column,
                "n": int(diff.size),
                "x": mean_axis,
                "diff": diff,
                "mean_diff": mean_diff,
                "scale": float(10 ** mean_diff) if bool(log) else float(np.exp(mean_diff)),
                "regression": regression,
                "x_lo": x_lo,
                "x_hi": x_hi,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "stats_loc": stats_loc,
            }

    row_n: dict[int, str] = {}
    for row_index, _ in enumerate(class_items):
        counts = [cache[(row_index, col)]["n"] for col in range(ncols) if cache.get((row_index, col)) is not None]
        if not counts:
            row_n[row_index] = "n=0"
        elif len(set(counts)) == 1:
            row_n[row_index] = f"n={counts[0]}"
        else:
            row_n[row_index] = f"n={min(counts)}-{max(counts)}"

    species_order = [
        f"{str(class_name).replace('_', ' ')} ({row_n[row_index]})"
        for row_index, (_, class_name) in enumerate(class_items)
    ]

    points_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    nodata_rows: list[dict[str, Any]] = []
    bounds_rows: list[dict[str, Any]] = []

    for row_index, _ in enumerate(class_items):
        species_label = species_order[row_index]
        for column_index, column in enumerate(selected_columns):
            column = str(column)
            descriptor_label = descriptor_label_map[column]
            info = cache.get((row_index, column_index))

            if info is None:
                x_lo, x_hi = -1.0, 1.0
                y_lo, y_hi = -1.0, 1.0
                bounds_rows.append({"species": species_label, "descriptor": descriptor_label, "x": x_lo, "y": y_lo})
                bounds_rows.append({"species": species_label, "descriptor": descriptor_label, "x": x_hi, "y": y_hi})
                nodata_rows.append(
                    {
                        "species": species_label,
                        "descriptor": descriptor_label,
                        "x": 0.5 * (x_lo + x_hi),
                        "y": 0.0,
                        "label": "No matched pairs",
                    }
                )
                continue

            mean_axis = np.asarray(info["x"], dtype=float)
            diff = np.asarray(info["diff"], dtype=float)
            mean_diff = float(info["mean_diff"])
            regression = info["regression"]
            x_lo = float(info["x_lo"])
            x_hi = float(info["x_hi"])
            y_lo = float(info["y_lo"])
            y_hi = float(info["y_hi"])

            bounds_rows.append({"species": species_label, "descriptor": descriptor_label, "x": x_lo, "y": y_lo})
            bounds_rows.append({"species": species_label, "descriptor": descriptor_label, "x": x_hi, "y": y_hi})

            points_rows.extend(
                {
                    "species": species_label,
                    "descriptor": descriptor_label,
                    "x": float(x_value),
                    "y": float(y_value),
                }
                for x_value, y_value in zip(mean_axis, diff)
            )

            line_rows.extend(
                [
                    {
                        "species": species_label,
                        "descriptor": descriptor_label,
                        "line_type": "Zero bias (0)",
                        "slope": 0.0,
                        "intercept": 0.0,
                    },
                    {
                        "species": species_label,
                        "descriptor": descriptor_label,
                        "line_type": "Mean bias",
                        "slope": 0.0,
                        "intercept": mean_diff,
                    },
                ]
            )

            if regression is not None:
                line_rows.append(
                    {
                        "species": species_label,
                        "descriptor": descriptor_label,
                        "line_type": "Trend vs size",
                        "slope": float(regression.slope),
                        "intercept": float(regression.intercept),
                    }
                )

            if panel_stats:
                slope_text = f"{regression.slope:.3g} (p={regression.pvalue:.2g})" if regression is not None else "n/a"
                location = info.get("stats_loc") or str(stats_box_loc)
                x_rel, y_rel, ha, va = stats_box_anchor(location)
                x_text = x_lo + x_rel * (x_hi - x_lo)
                y_text = y_lo + y_rel * (y_hi - y_lo)
                label = (
                    f"mean bias={mean_diff:.3g}\n"
                    f"scale={float(info['scale']):.3g}\n"
                    f"slope={slope_text}"
                )
                stats_rows.append(
                    {
                        "species": species_label,
                        "descriptor": descriptor_label,
                        "x": float(x_text),
                        "y": float(y_text),
                        "ha": str(ha),
                        "va": str(va),
                        "label": label,
                    }
                )

    points_df = pd.DataFrame(points_rows)
    line_df = pd.DataFrame(line_rows)
    bounds_df = pd.DataFrame(bounds_rows)
    nodata_df = pd.DataFrame(nodata_rows) if nodata_rows else pd.DataFrame(
        columns=["species", "descriptor", "x", "y", "label"]
    )
    stats_df = pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame(
        columns=["species", "descriptor", "x", "y", "ha", "va", "label"]
    )

    if points_df.empty and nodata_df.empty:
        raise ValueError("No matched pairs found for Bland-Altman plotting.")

    panel_order = [f"{species} | {descriptor}" for species in species_order for descriptor in descriptor_order]
    for frame_to_cast in (points_df, line_df, bounds_df, nodata_df, stats_df):
        if frame_to_cast.empty:
            continue
        frame_to_cast["species"] = pd.Categorical(
            frame_to_cast["species"],
            categories=species_order,
            ordered=True,
        )
        frame_to_cast["descriptor"] = pd.Categorical(
            frame_to_cast["descriptor"],
            categories=descriptor_order,
            ordered=True,
        )
        frame_to_cast["panel"] = pd.Categorical(
            frame_to_cast["species"].astype(str) + " | " + frame_to_cast["descriptor"].astype(str),
            categories=panel_order,
            ordered=True,
        )

    if not line_df.empty:
        line_df["line_type"] = pd.Categorical(
            line_df["line_type"],
            categories=["Mean bias", "Trend vs size", "Zero bias (0)"],
            ordered=True,
        )

    color_map = {
        "Mean bias": "#111827",
        "Zero bias (0)": "#dc2626",
        "Trend vs size": "#d97706",
    }
    linetype_map = {
        "Mean bias": "solid",
        "Zero bias (0)": "solid",
        "Trend vs size": "dashdot",
    }

    nrows = len(species_order)
    plot = (
        plotnine.ggplot()
        + plotnine.geom_blank(data=bounds_df, mapping=plotnine.aes(x="x", y="y"), inherit_aes=False)
        + plotnine.facet_wrap("~ panel", ncol=ncols, scales="free", drop=False)
        + plotnine.labs(
            title="Log-Bias Bland-Altman Grid" if bool(log) else "Bland-Altman Grid",
            x="mean(log10 GT, log10 Pred)" if bool(log) else "mean(ln GT, ln Pred)",
            y="log10(Pred / GT)" if bool(log) else "ln(Pred / GT)",
            color="",
            linetype="",
        )
        + plotnine.theme_bw()
        + plotnine.theme(
            figure_size=(4.5 * ncols, 3.0 * nrows),
            strip_text=plotnine.element_text(weight="bold"),
            plot_title=plotnine.element_text(weight="bold", ha="center"),
            legend_position="bottom",
            legend_direction="horizontal",
            axis_text_x=plotnine.element_text(size=8),
            axis_text_y=plotnine.element_text(size=8),
            subplots_adjust={"wspace": 0.12, "hspace": 0.12},
        )
    )

    if not points_df.empty:
        plot = plot + plotnine.geom_point(
            data=points_df,
            mapping=plotnine.aes(x="x", y="y"),
            color="#2a7fb8",
            alpha=0.30,
            size=0.8,
        )

    if not line_df.empty:
        plot = (
            plot
            + plotnine.geom_abline(
                data=line_df,
                mapping=plotnine.aes(
                    intercept="intercept",
                    slope="slope",
                    color="line_type",
                    linetype="line_type",
                ),
                size=0.8,
                show_legend=True,
            )
            + plotnine.scale_color_manual(values=color_map)
            + plotnine.scale_linetype_manual(values=linetype_map)
        )

    if not nodata_df.empty:
        plot = plot + plotnine.geom_text(
            data=nodata_df,
            mapping=plotnine.aes(x="x", y="y", label="label"),
            inherit_aes=False,
            size=12,
            color="#6b7280",
        )

    if panel_stats and not stats_df.empty:
        for ha, va in (("left", "top"), ("right", "top"), ("left", "bottom"), ("right", "bottom")):
            subset = stats_df[(stats_df["ha"] == ha) & (stats_df["va"] == va)]
            if subset.empty:
                continue
            plot = plot + plotnine.geom_text(
                data=subset,
                mapping=plotnine.aes(x="x", y="y", label="label"),
                inherit_aes=False,
                size=8.2,
                color="#334155",
                ha=ha,
                va=va,
            )

    _save_ggplot_if_requested(plot, save, "bland_altman.png")
    _show_plotnine(plot, show)
    return points_df, plot


def _draw_curved_rms_label(
    axis: Any,
    *,
    rms_value: float,
    axis_max: float,
    label: str = "CRMSD",
    fontsize: int = 9,
    color: str = "#444444",
) -> None:
    angles = np.linspace(np.deg2rad(125), np.deg2rad(90), 100)
    center_x, center_y = 1.0, 0.0
    x_arc = center_x + rms_value * np.cos(angles)
    y_arc = center_y + rms_value * np.sin(angles)

    mask = (x_arc >= 0) & (y_arc >= 0) & (np.sqrt(x_arc**2 + y_arc**2) <= axis_max)
    x_arc = x_arc[mask]
    y_arc = y_arc[mask]
    if len(x_arc) < 2:
        return

    indices = np.linspace(0, len(x_arc) - 1, len(label) + 2, dtype=int)[1:-1]
    for index, character in zip(indices, label):
        x_value = x_arc[index]
        y_value = y_arc[index]
        if index + 1 < len(x_arc):
            dx = x_arc[index + 1] - x_arc[index]
            dy = y_arc[index + 1] - y_arc[index]
        else:
            dx = x_arc[index] - x_arc[index - 1]
            dy = y_arc[index] - y_arc[index - 1]
        angle = np.degrees(np.arctan2(dy, dx))
        axis.text(
            x_value,
            y_value,
            character,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            rotation=angle,
            rotation_mode="anchor",
            fontweight="bold",
        )


def taylor_plot(
    data: Any,
    *,
    show: bool = True,
    align: str | None = "median",
    log: bool = True,
    columns: Sequence[str] | None = None,
    save: str | Path | None = None,
    class_names: Mapping[int | str, str] | None = None,
    exclude_classes: Sequence[int | str] | None = None,
    file_names: Sequence[str] | None = None,
    samples: Sequence[str | int] | None = None,
    sample_n: int | None = None,
    random_state: int | None = None,
) -> tuple[Any, Any | None]:

    sm = _require_skill_metrics()

    def safe_label(text: str) -> str:
        return str(text).replace("_", " ")

    frame, selected_columns, class_items = _processed_descriptor_frame(
        data,
        columns=columns,
        only_tp=True,
        log=False,
        align=align,
        class_names=class_names,
        exclude_classes=exclude_classes,
        file_names=file_names,
        samples=samples,
        sample_n=sample_n,
        random_state=random_state,
    )
    frame = frame[
        frame["status"].eq(STATUS_MATCHED) & frame["class_match"].eq(True)
    ].copy()
    if frame.empty:
        raise ValueError("No matched pairs found for Taylor plotting.")

    descriptor_label_map = {str(column): safe_label(str(column)) for column in selected_columns}
    descriptor_order = [descriptor_label_map[str(column)] for column in selected_columns]
    species_order = [str(name) for _, name in class_items]

    empty_columns = [
        "species",
        "descriptor",
        "descriptor_key",
        "n",
        "gt_std",
        "pred_std",
        "norm_std",
        "corr",
        "theta_rad",
        "theta_deg",
        "crmsd",
        "norm_crmsd",
        "valid",
        "reason",
        "scale",
    ]
    if not class_items:
        return pd.DataFrame(columns=empty_columns), None

    metrics_rows: list[dict[str, Any]] = []
    scale_label = "log10" if bool(log) else "linear"

    for class_id, class_name in class_items:
        species_label = str(class_name)
        subset = frame[
            frame["gt_class"].eq(float(class_id))
            & frame["pred_class"].eq(float(class_id))
        ].copy()

        for column in selected_columns:
            column = str(column)
            descriptor_label = descriptor_label_map[column]
            gt_column = f"gt_{column}"
            pred_column = f"pred_{column}"

            pair = subset[[gt_column, pred_column]].copy()
            pair = pair[np.isfinite(pair[gt_column]) & np.isfinite(pair[pred_column])]

            if bool(log):
                pair = pair[(pair[gt_column] > 0) & (pair[pred_column] > 0)]
                gt_values = (
                    np.log10(pair[gt_column].to_numpy(dtype=float))
                    if len(pair) else np.array([], dtype=float)
                )
                pred_values = (
                    np.log10(pair[pred_column].to_numpy(dtype=float))
                    if len(pair) else np.array([], dtype=float)
                )
            else:
                gt_values = pair[gt_column].to_numpy(dtype=float)
                pred_values = pair[pred_column].to_numpy(dtype=float)

            row = {
                "species": species_label,
                "descriptor": descriptor_label,
                "descriptor_key": column,
                "n": int(gt_values.size),
                "gt_std": np.nan,
                "pred_std": np.nan,
                "norm_std": np.nan,
                "corr": np.nan,
                "theta_rad": np.nan,
                "theta_deg": np.nan,
                "crmsd": np.nan,
                "norm_crmsd": np.nan,
                "valid": False,
                "reason": "",
                "scale": scale_label,
            }

            if gt_values.size < 2:
                row["reason"] = "Need at least 2 matched pairs"
                metrics_rows.append(row)
                continue

            gt_std = float(np.std(gt_values, ddof=1))
            pred_std = float(np.std(pred_values, ddof=1))
            row["gt_std"] = gt_std
            row["pred_std"] = pred_std

            if not np.isfinite(gt_std) or gt_std <= 0:
                row["reason"] = "GT standard deviation is zero"
                metrics_rows.append(row)
                continue

            if not np.isfinite(pred_std) or pred_std <= 0:
                row["reason"] = "Pred standard deviation is zero"
                metrics_rows.append(row)
                continue

            corr = float(np.corrcoef(gt_values, pred_values)[0, 1])
            if not np.isfinite(corr):
                row["reason"] = "Correlation is undefined"
                metrics_rows.append(row)
                continue

            corr = float(np.clip(corr, -1.0, 1.0))
            centered = (pred_values - float(np.mean(pred_values))) - (
                gt_values - float(np.mean(gt_values))
            )
            crmsd = float(np.sqrt(np.mean(centered**2)))
            norm_std = float(pred_std / gt_std)
            norm_crmsd = float(crmsd / gt_std)
            theta_rad = float(np.arccos(corr))

            row.update(
                {
                    "norm_std": norm_std,
                    "corr": corr,
                    "theta_rad": theta_rad,
                    "theta_deg": float(np.degrees(theta_rad)),
                    "crmsd": crmsd,
                    "norm_crmsd": norm_crmsd,
                    "valid": True,
                }
            )
            metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows, columns=empty_columns)
    if metrics_df.empty:
        raise ValueError("No data available for Taylor plotting.")

    metrics_df["species"] = pd.Categorical(metrics_df["species"], categories=species_order, ordered=True)
    metrics_df["descriptor"] = pd.Categorical(metrics_df["descriptor"], categories=descriptor_order, ordered=True)

    valid_df = metrics_df[metrics_df["valid"].eq(True)].copy()
    if valid_df.empty:
        raise ValueError("No valid matched pairs found for Taylor plotting.")

    max_norm_std = float(valid_df["norm_std"].max())
    max_norm_crmsd = float(valid_df["norm_crmsd"].max())

    axis_max = max(1.2, max_norm_std * 1.2, 1.0 + max_norm_crmsd)
    axis_max = float(np.ceil(axis_max / 0.1) * 0.1)

    radial_step = 0.15 if axis_max <= 1.5 else 0.2
    tick_std = np.arange(0.0, axis_max + radial_step, radial_step)
    tick_rms = np.arange(radial_step, axis_max + radial_step, radial_step)

    fig_cols = min(3, max(1, len(descriptor_order)))
    fig_rows = int(math.ceil(len(descriptor_order) / float(fig_cols)))

    figure = plt.figure(figsize=(5.2 * fig_cols, 4.9 * fig_rows + 1.0))
    cmap = plt.get_cmap("tab10", max(1, len(species_order)))
    color_lookup = {species: cmap(index) for index, species in enumerate(species_order)}

    for axis_index, descriptor_label in enumerate(descriptor_order, start=1):
        axis = figure.add_subplot(fig_rows, fig_cols, axis_index)

        descriptor_valid = valid_df[valid_df["descriptor"].astype(str).eq(descriptor_label)].copy()
        descriptor_invalid = metrics_df[
            metrics_df["descriptor"].astype(str).eq(descriptor_label)
            & ~metrics_df["valid"].eq(True)
        ].copy()

        if descriptor_valid.empty:
            axis.axis("off")
            axis.text(
                0.5,
                0.5,
                f"{descriptor_label}\n\nNo valid matched pairs",
                ha="center",
                va="center",
                fontsize=11,
                color="#6b7280",
                transform=axis.transAxes,
            )
            continue

        stds = [1.0]
        rmsds = [0.0]
        cors = [1.0]
        present_species: list[str] = []

        for species in species_order:
            row = descriptor_valid[descriptor_valid["species"].astype(str).eq(species)]
            if row.empty:
                continue
            stds.append(float(row["norm_std"].iloc[0]))
            rmsds.append(float(row["norm_crmsd"].iloc[0]))
            cors.append(float(row["corr"].iloc[0]))
            present_species.append(species)

        stds = np.asarray(stds, dtype=float)
        rmsds = np.asarray(rmsds, dtype=float)
        cors = np.asarray(cors, dtype=float)

        plt.sca(axis)
        sm.taylor_diagram(
            stds,
            rmsds,
            cors,
            axisMax=axis_max,
            tickSTD=tick_std,
            tickRMS=tick_rms,
            tickCOR=[0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
            colFrame="#222222",
            colSTD="#d0d0d0",
            colRMS="#444444",
            colCOR="#b7c3ff",
            styleSTD="-",
            styleRMS="--",
            styleCOR="-",
            widthSTD=0.8,
            widthRMS=1.0,
            widthCOR=0.8,
            markerDisplayed="marker",
            markerLabel=[""] * (1 + len(present_species)),
            markerLegend="off",
            markerColor="red",
            markerSymbol="o",
            markerSize=0,
            styleOBS="-",
            colOBS="#222222",
            markerObs="none",
            titleOBS="",
            titleSTD="on",
            titleRMS="off",
            titleCOR="on",
            rmsLabelFormat="0:.2f",
            showlabelsSTD="on",
            showlabelsRMS="on",
            showlabelsCOR="on",
            checkStats="off",
        )

        for species in present_species:
            row = descriptor_valid[descriptor_valid["species"].astype(str).eq(species)].iloc[0]
            sm.taylor_diagram(
                np.asarray([1.0, float(row["norm_std"])]),
                np.asarray([0.0, float(row["norm_crmsd"])]),
                np.asarray([1.0, float(row["corr"])]),
                overlay="on",
                markerDisplayed="marker",
                markerLegend="off",
                markerLabel=["", ""],
                markerColor=color_lookup[species],
                markerSymbol="o",
                markerSize=10,
                alpha=1.0,
                checkStats="off",
            )
            _draw_curved_rms_label(axis, rms_value=1.12, axis_max=axis_max)

        axis.set_title(descriptor_label, fontsize=12, fontweight="bold", pad=12)

        footer = f"{len(descriptor_valid)} valid species"
        if not descriptor_invalid.empty:
            footer += f" | {len(descriptor_invalid)} skipped"

        axis.text(
            0.03,
            0.03,
            footer,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#475569",
        )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6,
            label="Reference",
        )
    ]
    legend_handles.extend(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_lookup[species],
            markeredgecolor=color_lookup[species],
            markersize=6,
            label=species,
        )
        for species in species_order
    )

    figure.suptitle(
        f"Normalized Taylor Diagram ({scale_label})",
        y=0.98,
        fontsize=18,
        fontweight="bold",
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=10,
    )
    figure.tight_layout(rect=[0.02, 0.09, 0.98, 0.92])

    _save_matplotlib_if_requested(figure, save, "taylor_plot.png")
    _show_or_close_matplotlib(figure, show)
    return metrics_df, figure


def target_diagram(
    data: Any,
    *,
    show: bool = True,
    align: str | None = "median",
    log: bool = True,
    columns: Sequence[str] | None = None,
    panel_stats: bool = True,
    save: str | Path | None = None,
    class_names: Mapping[int | str, str] | None = None,
    exclude_classes: Sequence[int | str] | None = None,
    file_names: Sequence[str] | None = None,
    samples: Sequence[str | int] | None = None,
    sample_n: int | None = None,
    random_state: int | None = None,
    cuts: Sequence[float] | None = None,
    label_points: bool | None = None,
) -> tuple[Any, Any | None]:
    del panel_stats  # Kept for API compatibility with the class-based version.


    def format_tick(value: float) -> str:
        text = f"{value:.2f}"
        return text.rstrip("0").rstrip(".")

    frame, selected_columns, class_items = _processed_descriptor_frame(
        data,
        columns=columns,
        only_tp=True,
        log=False,
        align=align,
        class_names=class_names,
        exclude_classes=exclude_classes,
        file_names=file_names,
        samples=samples,
        sample_n=sample_n,
        random_state=random_state,
    )
    frame = frame[
        frame["status"].eq(STATUS_MATCHED) & frame["class_match"].eq(True)
    ].copy()
    if frame.empty:
        raise ValueError("No matched pairs found for target diagram plotting.")

    descriptor_label_map = {str(column): str(column).replace("_", " ") for column in selected_columns}
    descriptor_order = [descriptor_label_map[str(column)] for column in selected_columns]
    species_order = [str(name) for _, name in class_items]
    empty_columns = [
        "species",
        "descriptor",
        "descriptor_key",
        "n",
        "gt_std",
        "pred_std",
        "std_delta",
        "bias",
        "norm_bias",
        "crmsd",
        "norm_crmsd",
        "signed_norm_crmsd",
        "rmsd",
        "norm_rmsd",
        "valid",
        "reason",
        "scale",
    ]
    if not class_items:
        return pd.DataFrame(columns=empty_columns), None

    metrics_rows: list[dict[str, Any]] = []
    scale_label = "log10" if bool(log) else "linear"

    for class_id, class_name in class_items:
        species_label = str(class_name)
        subset = frame[
            frame["gt_class"].eq(float(class_id))
            & frame["pred_class"].eq(float(class_id))
        ].copy()

        for column in selected_columns:
            column = str(column)
            descriptor_label = descriptor_label_map[column]
            gt_column = f"gt_{column}"
            pred_column = f"pred_{column}"
            pair = subset[[gt_column, pred_column]].copy()
            pair = pair[np.isfinite(pair[gt_column]) & np.isfinite(pair[pred_column])]

            if bool(log):
                pair = pair[(pair[gt_column] > 0) & (pair[pred_column] > 0)]
                gt_values = np.log10(pair[gt_column].to_numpy(dtype=float)) if len(pair) else np.array([], dtype=float)
                pred_values = np.log10(pair[pred_column].to_numpy(dtype=float)) if len(pair) else np.array([], dtype=float)
            else:
                gt_values = pair[gt_column].to_numpy(dtype=float)
                pred_values = pair[pred_column].to_numpy(dtype=float)

            row = {
                "species": species_label,
                "descriptor": descriptor_label,
                "descriptor_key": column,
                "n": int(gt_values.size),
                "gt_std": np.nan,
                "pred_std": np.nan,
                "std_delta": np.nan,
                "bias": np.nan,
                "norm_bias": np.nan,
                "crmsd": np.nan,
                "norm_crmsd": np.nan,
                "signed_norm_crmsd": np.nan,
                "rmsd": np.nan,
                "norm_rmsd": np.nan,
                "valid": False,
                "reason": "",
                "scale": scale_label,
            }

            if gt_values.size < 2:
                row["reason"] = "Need at least 2 matched pairs"
                metrics_rows.append(row)
                continue

            gt_std = float(np.std(gt_values, ddof=1))
            pred_std = float(np.std(pred_values, ddof=1))
            row["gt_std"] = gt_std
            row["pred_std"] = pred_std
            row["std_delta"] = float(pred_std - gt_std)

            if not np.isfinite(gt_std) or gt_std <= 0:
                row["reason"] = "GT standard deviation is zero"
                metrics_rows.append(row)
                continue
            if not np.isfinite(pred_std) or pred_std <= 0:
                row["reason"] = "Pred standard deviation is zero"
                metrics_rows.append(row)
                continue

            diff = pred_values - gt_values
            bias_value = float(np.mean(diff))
            centered = (pred_values - float(np.mean(pred_values))) - (gt_values - float(np.mean(gt_values)))
            crmsd = float(np.sqrt(np.mean(centered**2)))
            rmsd = float(np.sqrt(np.mean(diff**2)))
            norm_bias = float(bias_value / gt_std)
            norm_crmsd = float(crmsd / gt_std)
            sign = 1.0 if pred_std >= gt_std else -1.0
            signed_norm_crmsd = float(sign * norm_crmsd)
            norm_rmsd = float(rmsd / gt_std)

            row.update(
                {
                    "bias": bias_value,
                    "norm_bias": norm_bias,
                    "crmsd": crmsd,
                    "norm_crmsd": norm_crmsd,
                    "signed_norm_crmsd": signed_norm_crmsd,
                    "rmsd": rmsd,
                    "norm_rmsd": norm_rmsd,
                    "valid": True,
                }
            )
            metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows, columns=empty_columns)
    if metrics_df.empty:
        raise ValueError("No data available for target diagram plotting.")

    metrics_df["species"] = pd.Categorical(metrics_df["species"], categories=species_order, ordered=True)
    metrics_df["descriptor"] = pd.Categorical(metrics_df["descriptor"], categories=descriptor_order, ordered=True)

    valid_df = metrics_df[metrics_df["valid"].eq(True)].copy()
    if valid_df.empty:
        raise ValueError("No valid matched pairs found for target diagram plotting.")

    max_extent = max(
        float(valid_df["norm_rmsd"].max()) * 1.12,
        float(np.abs(valid_df["norm_bias"]).max()) * 1.12,
        float(np.abs(valid_df["signed_norm_crmsd"]).max()) * 1.12,
    )
    if max_extent <= 0.30:
        grid_step = 0.05
    elif max_extent <= 0.60:
        grid_step = 0.10
    elif max_extent <= 1.20:
        grid_step = 0.20
    elif max_extent <= 2.0:
        grid_step = 0.25
    elif max_extent <= 4.0:
        grid_step = 0.5
    else:
        grid_step = 1.0

    axis_limit = float(np.ceil(max(max_extent, 2.0 * grid_step) / grid_step) * grid_step)
    circle_levels = np.arange(grid_step, axis_limit + 0.5 * grid_step, grid_step)

    cmap = plt.get_cmap("tab20", max(1, len(species_order)))
    color_lookup = {species: cmap(index) for index, species in enumerate(species_order)}
    if cuts is not None:
        circle_levels = np.asarray([float(value) for value in cuts], dtype=float)
        circle_levels = circle_levels[np.isfinite(circle_levels) & (circle_levels > 0)]
        circle_levels = np.unique(circle_levels)
        if circle_levels.size:
            axis_limit = float(max(axis_limit, np.max(circle_levels)))

    if label_points is None:
        label_points = False

    def decorate_target_axes(axis: Any, *, title: str | None = None) -> None:
        axis.set_xlim(-axis_limit, axis_limit)
        axis.set_ylim(-axis_limit, axis_limit)
        axis.set_aspect("equal", adjustable="box")
        axis.axhline(0.0, color="#94a3b8", linewidth=0.9, zorder=0)
        axis.axvline(0.0, color="#94a3b8", linewidth=0.9, zorder=0)
        axis.grid(color="#e5e7eb", linewidth=0.8)
        axis.set_xlabel("signed normalized centered RMSD", fontsize=9)
        axis.set_ylabel("normalized bias", fontsize=9)
        axis.tick_params(labelsize=8)
        if title is not None:
            axis.set_title(title, fontweight="bold", fontsize=11)

        for level in circle_levels:
            circle = plt.Circle(
                (0.0, 0.0),
                radius=float(level),
                fill=False,
                edgecolor="#cbd5e1",
                linestyle="--",
                linewidth=0.8,
                zorder=0,
            )
            axis.add_patch(circle)
            diagonal = float(level / np.sqrt(2.0))
            if diagonal <= axis_limit:
                axis.text(
                    diagonal,
                    diagonal,
                    format_tick(float(level)),
                    color="#94a3b8",
                    fontsize=7,
                    ha="left",
                    va="bottom",
                )

        axis.scatter(
            [0.0],
            [0.0],
            marker="+",
            s=80,
            color="#111827",
            linewidths=1.1,
            zorder=3,
        )

    fig_cols = min(3, max(1, len(descriptor_order)))
    fig_rows = int(np.ceil(len(descriptor_order) / float(fig_cols)))
    figure, axes = plt.subplots(
        fig_rows,
        fig_cols,
        figsize=(4.6 * fig_cols + 1.8, 4.0 * fig_rows + 0.8),
    )
    axes_array = np.atleast_1d(axes).ravel()

    for axis_index, descriptor_label in enumerate(descriptor_order):
        axis = axes_array[axis_index]
        decorate_target_axes(axis, title=descriptor_label)

        descriptor_valid = valid_df[valid_df["descriptor"].astype(str).eq(descriptor_label)]
        if descriptor_valid.empty:
            axis.text(
                0.5,
                0.5,
                "No valid\nmatched pairs",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="#6b7280",
            )
            continue

        for _, row in descriptor_valid.iterrows():
            species_label = str(row["species"])
            axis.scatter(
                float(row["signed_norm_crmsd"]),
                float(row["norm_bias"]),
                s=48,
                color=color_lookup[species_label],
                edgecolor="white",
                linewidth=0.7,
                marker="o",
                zorder=4,
            )
            if label_points:
                axis.annotate(
                    species_label,
                    xy=(float(row["signed_norm_crmsd"]), float(row["norm_bias"])),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color="#111827",
                )

    for axis in axes_array[len(descriptor_order):]:
        axis.remove()

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_lookup[species],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=8,
            label=species,
        )
        for species in species_order
    ]

    figure.suptitle(
        f"Normalized Target Diagram ({scale_label})",
        y=0.98,
        fontweight="bold",
        fontsize=14,
    )
    figure.legend(
        handles=legend_handles,
        title="Species",
        loc="center left",
        bbox_to_anchor=(0.88, 0.50),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    figure.tight_layout(rect=[0.0, 0.04, 0.86, 0.92])

    _save_matplotlib_if_requested(figure, save, "target_diagram.png")
    _show_or_close_matplotlib(figure, show)
    return metrics_df, figure


def target_plot(*args, **kwargs):
    """Alias for :func:`target_diagram`."""
    return target_diagram(*args, **kwargs)


def per_sample_abundance(
    data: Any,
    *,
    conf: float | None = None,
    columns: Sequence[str] | None = None,
    only_tp: bool = True,
    iou_threshold: float = 0.7,
    align: str | None = "median",
    log10: bool = False,
    out_dir: str | Path = ".",
    show: bool = True,
    class_names: Mapping[int | str, str] | None = None,
    exclude_classes: Sequence[int | str] | None = None,
    file_names: Sequence[str] | None = None,
    samples: Sequence[str | int] | None = None,
    sample_n: int | None = None,
    random_state: int | None = None,
    save_csv: bool = False,
    filename: str = "per_sample_abundance.csv",
):
    del conf, iou_threshold, show
 
    frame, _, class_items = _processed_descriptor_frame(
        data,
        columns=columns,
        only_tp=bool(only_tp),
        log=bool(log10),
        align=align,
        class_names=class_names,
        exclude_classes=exclude_classes,
        file_names=file_names,
        samples=samples,
        sample_n=sample_n,
        random_state=random_state,
    )

    if frame.empty:
        output = pd.DataFrame(columns=["sample", "total_gt", "total_pred"])
    else:
        class_ids = [class_id for class_id, _ in class_items]
        keep_gt = [float(class_id) for class_id in class_ids]
        rows: list[dict[str, Any]] = []
        for sample in sorted(frame["sample"].dropna().astype(str).unique().tolist()):
            subset = frame[frame["sample"].eq(sample)]
            n_gt = int(
                (
                    subset["gt_class"].isin(keep_gt)
                    & subset["status"].isin([STATUS_MATCHED, STATUS_MISSED_GT])
                ).sum()
            )
            n_pred = int(
                (
                    subset["pred_class"].isin(keep_gt)
                    & subset["status"].isin([STATUS_MATCHED, STATUS_EXTRA_PRED])
                ).sum()
            )
            rows.append({"sample": sample, "total_gt": n_gt, "total_pred": n_pred})
        output = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)

    if save_csv:
        output_path = Path(out_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)

    return output


def plot_per_sample_abundance(
    data: Any,
    *,
    show: bool = True,
    save: str | Path | None = None,
    class_names: Mapping[int | str, str] | None = None,
    exclude_classes: Sequence[int | str] | None = None,
    file_names: Sequence[str] | None = None,
    samples: Sequence[str | int] | None = None,
    sample_n: int | None = None,
    random_state: int | None = None,
    **kwargs,
) -> tuple[Any, Any | None]:
    plotnine = _require_plotnine()


    kwargs = dict(kwargs)
    kwargs.pop("show", None)
    kwargs.setdefault("only_tp", False)
    kwargs.setdefault("align", None)

    table = per_sample_abundance(
        data,
        show=False,
        class_names=class_names,
        exclude_classes=exclude_classes,
        file_names=file_names,
        samples=samples,
        sample_n=sample_n,
        random_state=random_state,
        **kwargs,
    )
    if table.empty:
        return table, None

    frame, _, class_items = _processed_descriptor_frame(
        data,
        columns=kwargs.get("columns"),
        only_tp=bool(kwargs.get("only_tp", False)),
        log=bool(kwargs.get("log10", False)),
        align=kwargs.get("align", None),
        class_names=class_names,
        exclude_classes=exclude_classes,
        file_names=file_names,
        samples=samples,
        sample_n=sample_n,
        random_state=random_state,
    )
    class_map = {float(class_id): str(name) for class_id, name in class_items}

    dominant_rows: list[dict[str, Any]] = []
    if not frame.empty:
        for sample in table["sample"].astype(str).tolist():
            subset = frame[frame["sample"].astype(str).eq(sample)]
            gt_subset = subset[
                subset["status"].isin([STATUS_MATCHED, STATUS_MISSED_GT])
                & subset["gt_class"].notna()
            ]
            if gt_subset.empty:
                dominant_rows.append({"sample": sample, "dominant_species": "unknown"})
                continue
            value_counts = gt_subset["gt_class"].astype(float).value_counts()
            class_id = float(value_counts.index[0])
            dominant_rows.append(
                {
                    "sample": sample,
                    "dominant_species": class_map.get(class_id, str(int(class_id))),
                }
            )

    dominant_df = pd.DataFrame(dominant_rows) if dominant_rows else pd.DataFrame(
        columns=["sample", "dominant_species"]
    )
    plot_df = table.merge(dominant_df, on="sample", how="left")
    plot_df["dominant_species"] = plot_df["dominant_species"].fillna("unknown").astype(str)

    r_text = "n/a"
    x_values = plot_df["total_gt"].to_numpy(dtype=float)
    y_values = plot_df["total_pred"].to_numpy(dtype=float)
    if x_values.size >= 2 and np.unique(x_values).size >= 2 and np.unique(y_values).size >= 2:
        r_text = f"{float(np.corrcoef(x_values, y_values)[0, 1]):.2f}"

    x_min = float(np.min(x_values)) if x_values.size else 0.0
    x_max = float(np.max(x_values)) if x_values.size else 1.0
    y_min = float(np.min(y_values)) if y_values.size else 0.0
    y_max = float(np.max(y_values)) if y_values.size else 1.0
    lo = min(x_min, y_min)
    hi = max(x_max, y_max)
    pad = 0.06 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad

    plot = (
        plotnine.ggplot(plot_df, plotnine.aes(x="total_gt", y="total_pred", color="dominant_species"))
        + plotnine.geom_blank(
            data=pd.DataFrame({"x": [lo, hi], "y": [lo, hi]}),
            mapping=plotnine.aes(x="x", y="y"),
            inherit_aes=False,
        )
        + plotnine.geom_abline(intercept=0.0, slope=1.0, linetype="dashed", color="#374151", size=0.6)
        + plotnine.geom_point(size=2.2, alpha=0.95)
        + plotnine.labs(
            title=f"Total abundance per sample (r={r_text})",
            x="Manual abundance",
            y="Model predictions abundance",
            color="Species",
        )
        + plotnine.theme_bw()
        + plotnine.theme(
            figure_size=(5.4, 4.6),
            axis_text_x=plotnine.element_text(size=10),
            axis_text_y=plotnine.element_text(size=8),
            plot_title=plotnine.element_text(weight="bold", ha="center"),
            legend_position=(0.18, 0.80),
            legend_direction="vertical",
            legend_title=plotnine.element_text(size=9),
            legend_text=plotnine.element_text(size=8),
        )
    )

    _save_ggplot_if_requested(plot, save, "per_sample_abundance.png")
    _show_plotnine(plot, show)
    return table, plot

