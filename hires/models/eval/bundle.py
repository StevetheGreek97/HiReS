from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, List, Optional

import pandas as pd

from .pair import Pair


@dataclass
class Bundle:
    """Simple container for `Pair` objects with add/extend helpers and small I/O utils."""
    pairs: List[Pair] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Iterator[Pair]:
        return iter(self.pairs)

    def __getitem__(self, idx: int | slice) -> Pair | list[Pair]:
        return self.pairs[idx]

    def add(self, pair: Pair) -> None:
        """Append a single Pair."""
        self.pairs.append(pair)

    def extend(self, items: Iterable[Pair]) -> None:
        """Extend with an iterable of Pair objects."""
        self.pairs.extend(items)

    def filter(self, fn: Callable[[Pair], bool]) -> "Bundle":
        """Return a new Bundle containing only pairs where `fn(pair)` is True."""
        return Bundle([p for p in self.pairs if fn(p)])

    def _get_field(self, pair: Pair, key: str) -> Any:
        """Try to extract a field from a Pair via attribute or `to_dict()` fallback."""
        try:
            return getattr(pair, key)
        except Exception:
            try:
                return pair.to_dict().get(key)
            except Exception:
                return None

    def filter_by_iou(
        self, *, min_iou: Optional[float] = None, max_iou: Optional[float] = None
    ) -> "Bundle":
        """Filter pairs by an `iou` value (looks for attribute or record key 'iou').

        Returns pairs with min_iou <= iou <= max_iou. Use None to omit a bound.
        """
        def pred(p: Pair) -> bool:
            val = self._get_field(p, "iou")
            if val is None:
                return False
            try:
                v = float(val)
            except Exception:
                return False
            if min_iou is not None and v < min_iou:
                return False
            if max_iou is not None and v > max_iou:
                return False
            return True

        return self.filter(pred)

    def filter_by_status(self, statuses: str | Iterable[str]) -> "Bundle":
        """Filter pairs by `status` value (attribute or record key).

        `statuses` may be a single status string or an iterable of allowed statuses.
        """
        if isinstance(statuses, str):
            allowed = {statuses}
        else:
            allowed = set(statuses)

        def pred(p: Pair) -> bool:
            val = self._get_field(p, "status")
            return val in allowed

        return self.filter(pred)

    def to_records(self) -> List[dict[str, Any]]:
        return [p.to_dict() for p in self.pairs]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_records())

    def to_csv(self, path: str, *, index: bool = False, **kwargs: Any) -> None:
        self.to_dataframe().to_csv(path, index=index, **kwargs)