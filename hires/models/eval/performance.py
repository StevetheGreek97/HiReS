from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Dict, Any
import numpy as np
import pandas as pd

from ..album import Album
from ..collection import Collection
from .match_maker import CollectionMatchMaker


@dataclass
class Performance:
    gt_album: Album
    pred_album: Album

    @staticmethod
    def _average_precision(recalls: Iterable[float], precisions: Iterable[float]) -> float:
        recalls = np.asarray(list(recalls), dtype=float)
        precisions = np.asarray(list(precisions), dtype=float)
        if recalls.size == 0:
            return 0.0
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def _collect_pairs_df(self, iou_threshold: float, *, treat_left_as_gt: bool = True) -> pd.DataFrame:
        """
        Build a DataFrame of pair outcomes across all matched collections for a given IoU threshold.

        If `treat_left_as_gt` is True we call CollectionMatchMaker(gt, pred).
        If False we call CollectionMatchMaker(pred, gt) — useful for AP calculation (predictions must be left).
        """
        left_album = self.gt_album if treat_left_as_gt else self.pred_album
        right_album = self.pred_album if treat_left_as_gt else self.gt_album

        left_map = left_album.name_map
        right_map = right_album.name_map

        matched_names = sorted(set(left_map.keys()) & set(right_map.keys()))
        rows: List[Dict[str, Any]] = []

        for name in matched_names:
            left_col: Collection = left_map[name]
            right_col: Collection = right_map[name]
            cm = CollectionMatchMaker(left_col, right_col, iou_threshold=float(iou_threshold))
            for p in cm.pairs_list():
                left_class = p.left_ann.class_id if p.left_ann is not None else None
                right_class = p.right_ann.class_id if p.right_ann is not None else None
                left_conf = getattr(p.left_ann, "confidence", None) if p.left_ann is not None else None
                rows.append(
                    {
                        "collection_name": name,
                        "left_class": left_class,
                        "right_class": right_class,
                        "status": p.status,
                        "iou": float(p.iou),
                        "left_conf": left_conf,
                    }
                )

        return pd.DataFrame(rows)

    def confusion_matrix(self, iou_threshold: float = 0.5, background: str = "background") -> pd.DataFrame:
        """
        Confusion matrix where rows = GT class, cols = Pred class.
        Uses CollectionMatchMaker(gt, pred, iou_threshold).
        """
        df = self._collect_pairs_df(iou_threshold, treat_left_as_gt=True)
        # fill None with background label so unmatched appear in matrix
        df["left_class"] = df["left_class"].fillna(background)
        df["right_class"] = df["right_class"].fillna(background)
        labels = sorted(set(df["left_class"].unique()) | set(df["right_class"].unique()), key=lambda x: str(x))
        cm = pd.crosstab(df["left_class"], df["right_class"])
        return cm.reindex(index=labels, columns=labels, fill_value=0)

    def _per_class_counts(self, iou_threshold: float = 0.5) -> pd.DataFrame:
        """
        Compute tp/fp/fn per class using CollectionMatchMaker(gt, pred, iou_threshold).
        """
        df = self._collect_pairs_df(iou_threshold, treat_left_as_gt=True)
        classes = sorted(set(df["left_class"].dropna().unique()) | set(df["right_class"].dropna().unique()))
        rows = []
        for cid in classes:
            tp = int(((df["status"] == "tp") & (df["left_class"] == cid) & (df["right_class"] == cid)).sum())
            # left=GT: "fp" = unmatched GT (missed detection) → FN for this class
            fn = int((((df["status"] == "fp") | (df["status"] == "misclassified")) & (df["left_class"] == cid)).sum())
            # right=pred: "fn" = unmatched pred (spurious prediction) → FP for this class
            fp = int((((df["status"] == "fn") | (df["status"] == "misclassified")) & (df["right_class"] == cid)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            rows.append({"class_id": cid, "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1})
        return pd.DataFrame(rows)

    def map_at(self, iou_threshold: float = 0.5) -> pd.DataFrame:
        """
        Compute AP per class at the given IoU threshold.
        For AP we treat predictions as the left side so we call CollectionMatchMaker(pred, gt, iou_threshold).
        """
        df = self._collect_pairs_df(iou_threshold, treat_left_as_gt=False)
        if df.empty:
            return pd.DataFrame(columns=["class_id", "mAP"])
        classes = sorted(set(df["left_class"].dropna().unique()) | set(df["right_class"].dropna().unique()))
        ap_rows = []
        for cid in classes:
            preds = df[df["left_class"] == cid].copy()
            n_gt = int((df["right_class"] == cid).sum())
            if n_gt == 0:
                ap_rows.append({"class_id": cid, "mAP": 0.0})
                continue
            if preds.empty:
                ap_rows.append({"class_id": cid, "mAP": 0.0})
                continue
            # sort by prediction confidence (desc); if no confidence available treat as 0
            preds["left_conf_f"] = preds["left_conf"].fillna(0.0).astype(float)
            preds = preds.sort_values("left_conf_f", ascending=False)
            tp = ((preds["status"] == "tp") & (preds["right_class"] == cid) & (preds["left_class"] == cid)).astype(int)
            fp = 1 - tp
            tp_cum = tp.cumsum()
            fp_cum = fp.cumsum()
            recall = tp_cum / n_gt
            precision = tp_cum / (tp_cum + fp_cum)
            ap = float(self._average_precision(recall.values, precision.values))
            ap_rows.append({"class_id": cid, "mAP": ap})
        return pd.DataFrame(ap_rows)

    def per_class_report(self, iou_threshold: float = 0.5, map_thresholds: Iterable[float] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with per-class: tp, fp, fn, precision, recall, f1, mAP@0.5, mAP@0.5-0.95.

        - `iou_threshold` determines base TP/FP/FN (usually 0.5).
        - `map_thresholds` is an iterable of IoU thresholds to average for mAP range; default 0.5..0.95 step 0.05.
        """
        if map_thresholds is None:
            map_thresholds = np.arange(0.5, 0.95 + 1e-8, 0.05)

        base = self._per_class_counts(iou_threshold=iou_threshold).set_index("class_id")
        map50 = self.map_at(0.5).set_index("class_id").rename(columns={"mAP": "mAP@0.5"})
        # compute mAP for the range and average
        all_maps = []
        for thr in map_thresholds:
            dfm = self.map_at(float(thr)).set_index("class_id").rename(columns={"mAP": f"mAP@{thr:.2f}"})
            all_maps.append(dfm)
        if all_maps:
            maps_merged = pd.concat(all_maps, axis=1).fillna(0.0)
            maps_merged["mAP@0.5-0.95"] = maps_merged.mean(axis=1)
        else:
            maps_merged = pd.DataFrame(columns=["mAP@0.5-0.95"])

        result = base.join(map50, how="outer").join(maps_merged[["mAP@0.5-0.95"]], how="outer")
        result = result.fillna(0.0).reset_index().rename(columns={"index": "class_id"})

        # add 'all' summary row
        all_row = {
            "class_id": "all",
            "tp": int(result["tp"].sum()),
            "fp": int(result["fp"].sum()),
            "fn": int(result["fn"].sum()),
        }
        all_row["precision"] = all_row["tp"] / (all_row["tp"] + all_row["fp"]) if (all_row["tp"] + all_row["fp"]) > 0 else 0.0
        all_row["recall"] = all_row["tp"] / (all_row["tp"] + all_row["fn"]) if (all_row["tp"] + all_row["fn"]) > 0 else 0.0
        all_row["f1"] = (2 * all_row["precision"] * all_row["recall"] / (all_row["precision"] + all_row["recall"])) if (all_row["precision"] + all_row["recall"]) > 0 else 0.0
        all_row["mAP@0.5"] = float(result["mAP@0.5"].mean()) if "mAP@0.5" in result.columns else 0.0
        all_row["mAP@0.5-0.95"] = float(result["mAP@0.5-0.95"].mean()) if "mAP@0.5-0.95" in result.columns else 0.0

        return pd.concat([pd.DataFrame([all_row]), result], ignore_index=True)