from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, List

import numpy as np
import pandas as pd

from .pair import Pair
from .iou import IoUMatrix
from .bundle import Bundle


from ..collection import Collection
from ..album import Album


class PairListAdapter:
    """Wrap a list[Pair] to provide `to_records()` and iteration like MatchMaker."""
    def __init__(self, pairs: List[Pair], left_collection: Collection | None = None, right_collection: Collection | None = None):
        self.pairs = pairs
        self.left_collection = left_collection
        self.right_collection = right_collection

    def to_records(self) -> List[dict[str, Any]]:
        return [p.to_dict() for p in self.pairs]

    def __iter__(self) -> Iterator[Pair]:
        return iter(self.pairs)


@dataclass
class CollectionMatchMaker:
    left: Collection
    right: Collection
    iou_threshold: float = 0.5
    class_aware: bool = False

    pairs: List[Pair] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        iou = IoUMatrix(self.left, self.right, return_dense=True)
        self._left_annotations = iou.left_annotations
        self._right_annotations = iou.right_annotations
        self._dense = iou.require_dense()
        self._build_pairs()

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[Pair]:
        return iter(self.pairs)

    def __getitem__(self, idx: int | slice) -> Pair | list[Pair]:
        return self.pairs[idx]

    def get_left(self, index: int) -> List[Pair]:
        return [p for p in self.pairs if getattr(p, "left_index", None) == index or getattr(p, "left_ann", None) == index]

    def get_right(self, index: int) -> List[Pair]:
        return [p for p in self.pairs if getattr(p, "right_index", None) == index or getattr(p, "right_ann", None) == index]

    def _status_for_pair(self, left_index: int, right_index: int) -> str:
        left_ann = self._left_annotations[left_index]
        right_ann = self._right_annotations[right_index]
        return "tp" if left_ann.class_id == right_ann.class_id else "misclassified"

    def _greedy_pairs(self) -> List[tuple[int, int, float]]:
        if self._dense.size == 0:
            return []

        left_idx, right_idx = np.where(self._dense >= self.iou_threshold)
        ious = self._dense[left_idx, right_idx]
        order = np.argsort(ious)[::-1]

        used_left: set[int] = set()
        used_right: set[int] = set()
        pairs: List[tuple[int, int, float]] = []

        for idx in order:
            li = int(left_idx[idx])
            ri = int(right_idx[idx])
            iou = float(ious[idx])

            if li in used_left or ri in used_right:
                continue

            status = self._status_for_pair(li, ri)
            if self.class_aware and status == "misclassified":
                continue

            used_left.add(li)
            used_right.add(ri)
            pairs.append((li, ri, iou))

        return pairs

    def _make_outcome(self, *, left_index: int | None, right_index: int | None, iou: float, status: str) -> Pair:
        return Pair.from_annotations(
            left_index=left_index,
            right_index=right_index,
            iou=iou,
            status=status,
            left_ann=self._left_annotations,
            right_ann=self._right_annotations,
        )

    def _build_pairs(self) -> None:
        raw_pairs = self._greedy_pairs()

        used_left = {l for l, _, _ in raw_pairs}
        used_right = {r for _, r, _ in raw_pairs}

        outcomes: List[Pair] = []

        for li, ri, iou in raw_pairs:
            outcomes.append(self._make_outcome(left_index=li, right_index=ri, iou=iou, status=self._status_for_pair(li, ri)))

        for li in range(len(self._left_annotations)):
            if li not in used_left:
                outcomes.append(self._make_outcome(left_index=li, right_index=None, iou=0.0, status="fp"))

        for ri in range(len(self._right_annotations)):
            if ri not in used_right:
                outcomes.append(self._make_outcome(left_index=None, right_index=ri, iou=0.0, status="fn"))

        self.pairs = outcomes

    def pairs_list(self) -> List[Pair]:
        return list(self.pairs)

    def to_records(self) -> List[dict[str, Any]]:
        return [p.to_dict() for p in self.pairs]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_records())

    def to_bundle(self) -> Bundle:
        return Bundle(pairs=self.pairs_list())

@dataclass
class AlbumMatchMaker:
    left: Album
    right: Album
    iou_threshold: float = 0.5
    class_aware: bool = False
    strict: bool = False

    def bundle(self) -> Bundle:
        left_map = self.left.name_map
        right_map = self.right.name_map

        left_names = set(left_map.keys())
        right_names = set(right_map.keys())

        matched_names = sorted(left_names & right_names)
        only_in_left = sorted(left_names - right_names)
        only_in_right = sorted(right_names - left_names)

        if self.strict and (only_in_left or only_in_right):
            raise ValueError(f"Albums do not contain the same collection names. Only in left: {only_in_left}. Only in right: {only_in_right}.")

        matched: dict[str, PairListAdapter] = {}
        bundle = Bundle()

        for name in matched_names:
            cm = CollectionMatchMaker(
                left_map[name],
                right_map[name],
                iou_threshold=self.iou_threshold,
                class_aware=self.class_aware,
            )
            adapter = PairListAdapter(cm.pairs_list(), left_collection=left_map[name], right_collection=right_map[name])
            matched[name] = adapter
            # extend returned bundle with all pairs from this collection match
            bundle.extend(adapter.pairs)


        return bundle