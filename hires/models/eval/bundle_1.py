#from __future__ import annotations

#from dataclasses import dataclass, field
#from pathlib import Path
#from typing import Any, Iterator
#
#import pandas as pd

#from .match_maker import MatchMaker


#@dataclass
#class Bundle:
#    matched: dict[str, MatchMaker] = field(default_factory=dict)
#    only_in_self: list[str] = field(default_factory=list)
#    only_in_other: list[str] = field(default_factory=list)
#    self_album_name: str | None = None
#    other_album_name: str | None = None
#
#    def __len__(self) -> int:
#        return len(self.matched)
#
#    def __iter__(self) -> Iterator[MatchMaker]:
#        return iter(self.matched.values())
#
#    def __getitem__(self, key: str) -> MatchMaker:
#        return self.matched[key]
#
#    @property
#    def matched_names(self) -> list[str]:
#        return sorted(self.matched.keys())
#
#    @property
#    def is_perfect_match(self) -> bool:
#        return not self.only_in_self and not self.only_in_other
#
#    def summary(self) -> dict[str, Any]:
#        return {
#            "self_album_name": self.self_album_name,
#            "other_album_name": self.other_album_name,
#            "n_matched": len(self.matched),
#            "n_only_in_self": len(self.only_in_self),
#            "n_only_in_other": len(self.only_in_other),
#            "matched_names": self.matched_names,
#            "only_in_self": sorted(self.only_in_self),
#            "only_in_other": sorted(self.only_in_other),
#            "is_perfect_match": self.is_perfect_match,
#        }
#
#    def to_records(self, include_unmatched: bool = True) -> list[dict[str, Any]]:
#        records: list[dict[str, Any]] = []
#
#        for collection_name, matchmaker in self.matched.items():
#            for row in matchmaker.to_records():
#                row["collection_name"] = collection_name
#                row["self_album_name"] = self.self_album_name
#                row["other_album_name"] = self.other_album_name
#                records.append(row)
#
#        if include_unmatched:
#            for name in self.only_in_self:
#                records.append(
#                    {
#                        "collection_name": name,
#                        "status": "only_in_self",
#                        "self_album_name": self.self_album_name,
#                        "other_album_name": self.other_album_name,
#                    }
#                )
#
#            for name in self.only_in_other:
#                records.append(
#                    {
#                        "collection_name": name,
#                        "status": "only_in_other",
#                        "self_album_name": self.self_album_name,
#                        "other_album_name": self.other_album_name,
#                    }
#                )
#
#        return records
#
#    def to_dataframe(self, include_unmatched: bool = True) -> pd.DataFrame:
#        return pd.DataFrame(self.to_records(include_unmatched=include_unmatched))
#
#    def to_csv(
#        self,
#        path: str | Path,
#        *,
#        include_unmatched: bool = True,
#        index: bool = False,
#        **kwargs: Any,
#    ) -> None:
#        self.to_dataframe(include_unmatched=include_unmatched).to_csv(
#            path,
#            index=index,
#            **kwargs,
#        )