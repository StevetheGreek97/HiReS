from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

OutcomeStatus = Literal["tp", "fn", "fp", "misclassified"]


@dataclass(frozen=True)
class Pair:
    left_ann: int | None
    right_ann: int | None
    iou: float
    status: OutcomeStatus

    @property
    def class_match(self) -> bool:
        return (
            self.left_ann.class_id is not None
            and self.right_ann.class_id is not None
            and self.left_ann.class_id == self.right_ann.class_id
        )

    @property
    def is_tp(self) -> bool:
        return self.status == "tp"

    @property
    def is_fp(self) -> bool:
        return self.status in {"fp", "misclassified"}

    @property
    def is_fn(self) -> bool:
        return self.status in {"fn", "misclassified"}
    @classmethod
    def from_annotations(
        cls,
        *,
        left_index: int | None,
        right_index: int | None,
        iou: float,
        status: OutcomeStatus,
        left_ann: list[Any],
        right_ann: list[Any],
    ) -> "Pair":
        left_ann = left_ann[left_index] if left_index is not None else None
        right_ann = right_ann[right_index] if right_index is not None else None

        return cls(
            left_ann=left_ann,
            right_ann=right_ann,
            iou=float(iou),
            status=status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_ann": self.left_ann,
            "right_ann": self.right_ann,
            "iou": self.iou,
            "status": self.status,
        }