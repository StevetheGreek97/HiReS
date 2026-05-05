from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Counter, Iterable, Iterator

#from models.eval.match_maker import MatchMaker
from .collection import Collection
from pathlib import Path
import pandas as pd

#from .eval.bundle_1 import Bundle

@dataclass
class Album:
    """
    Container for multiple Collection objects.

    Collections are matched one-to-one by `collection_name`.
    All collection names must be unique and non-empty.
    """

    collections: list[Collection] = field(default_factory=list)
    album_name: str | None = None

    def __post_init__(self) -> None:
        self._validate_unique_names()

    def __len__(self) -> int:
        return len(self.collections)

    def __iter__(self) -> Iterator[Collection]:
        return iter(self.collections)

    def __getitem__(self, key: int | slice | str) -> Collection | list[Collection]:
        if isinstance(key, str):
            for collection in self.collections:
                if collection.collection_name == key:
                    return collection
            raise KeyError(f"No collection found with name '{key}'")
        return self.collections[key]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return any(c.collection_name == item for c in self.collections)
        return item in self.collections

    def __repr__(self) -> str:
        return (
            f"Album("
            f"album_name={self.album_name!r}, "
            f"n_collections={len(self.collections)})"
        )

    def __str__(self) -> str:
        return (
            f"Album\n"
            f"  album_name    : {self.album_name}\n"
            f"  n_collections : {len(self.collections)}"
        )

    @property
    def names(self) -> list[str]:
        return [c.collection_name for c in self.collections if c.collection_name is not None]

    @property
    def name_map(self) -> dict[str, Collection]:
        return {
            c.collection_name: c
            for c in self.collections
            if c.collection_name is not None
        }

    def _validate_unique_names(self) -> None:
        seen: set[str] = set()

        for collection in self.collections:
            name = collection.collection_name

            if name is None or name == "":
                raise ValueError(
                    "All collections in an Album must have a non-empty collection_name."
                )

            if name in seen:
                raise ValueError(f"Duplicate collection_name '{name}' in Album.")

            seen.add(name)

    def add(self, collection: Collection) -> None:
        name = collection.collection_name

        if name is None or name == "":
            raise ValueError(
                "Collection must have a non-empty collection_name to be added to an Album."
            )

        if name in self:
            raise ValueError(f"Duplicate collection_name '{name}' in Album.")

        self.collections.append(collection)

    def extend(self, collections: Iterable[Collection]) -> None:
        for collection in collections:
            self.add(collection)

    def remove(self, collection_name: str) -> None:
        for i, collection in enumerate(self.collections):
            if collection.collection_name == collection_name:
                self.collections.pop(i)
                return
        raise KeyError(f"No collection found with name '{collection_name}'")

    def get(self, collection_name: str, default: Any = None) -> Collection | Any:
        for collection in self.collections:
            if collection.collection_name == collection_name:
                return collection
        return default

    def filter(self, names: Iterable[str]) -> "Album":
        names_set = set(names)
        return Album(
            collections=[c for c in self.collections if c.collection_name in names_set],
            album_name=self.album_name,
        )

    def summary(self) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []

        for collection in self.collections:
            summary.append(
                {
                    "album_name": self.album_name,
                    "collection_name": collection.collection_name,
                    "n_annotations": len(collection),
                    "image_path": str(collection.image_path) if collection.image_path else None,
                    "dpi": collection.dpi,
                    "unit": collection.unit,
                }
            )

        return summary
    
    def to_records(self) -> list[dict[str, Any]]:
        """
        Return ALL annotations across all collections (flattened).
        One row per annotation.
        """
        records: list[dict[str, Any]] = []

        for collection in self.collections:
            collection_records = collection.to_records()

            for row in collection_records:
                row["album_name"] = self.album_name
                records.append(row)

        return records
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.to_records())

    def to_csv(self, path: str, *, index: bool = False, **kwargs: Any) -> None:
        self.to_dataframe().to_csv(path, index=index, **kwargs)
    
    def set_scale(
        self,
        dpi: float | None = None,
        unit: str | None = None,
    ) -> None:
        """
        Apply scaling to ALL collections and their annotations.
        """
        for collection in self.collections:
            collection.set_scale(dpi=dpi, unit=unit)
    
    @classmethod
    def from_paths(
        cls,
        txt_paths: list[Path],
        *,
        image_dir: str | Path | None = None,
        image_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        album_name: str | None = None,
    ) -> "Album":
        collections: list[Collection] = []

        if image_dir is not None:
            image_dir = Path(image_dir)

        for txt_file in txt_paths:
            txt_file = Path(txt_file)
            name = txt_file.stem
            image_path = None

            if image_dir is not None:
                for ext in image_exts:
                    candidate = image_dir / f"{name}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break

            collection = Collection.read_txt(
                txt_file,
                collection_name=name,
                image_path=image_path,
            )
            collections.append(collection)

        return cls(collections=collections, album_name=album_name)

    @classmethod
    def from_dir(
        cls,
        txt_dir: str | Path,
        *,
        image_dir: str | Path | None = None,
        image_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        album_name: str | None = None,
    ) -> "Album":
        txt_dir = Path(txt_dir)
        if not txt_dir.exists():
            raise FileNotFoundError(f"{txt_dir} does not exist")

        collections: list[Collection] = []

        if image_dir is not None:
            image_dir = Path(image_dir)

        for txt_file in sorted(txt_dir.glob("*.txt")):
            name = txt_file.stem
            image_path = None

            if image_dir is not None:
                for ext in image_exts:
                    candidate = image_dir / f"{name}{ext}"
                    if candidate.exists():
                        image_path = candidate
                        break

            collection = Collection.read_txt(
                txt_file,
                collection_name=name,
                image_path=image_path,
            )
            collections.append(collection)

        return cls(
            collections=collections,
            album_name=album_name or txt_dir.name,
        )

    def save_crops(
        self,
        out_dir: str | Path,
        *,
        use_mask: bool = True,
        padding: int = 0,
        ext: str = "png",
    ) -> dict[str, list[str]]:
        """
        Save crops for all collections in the album.

        Each collection gets its own subfolder inside `out_dir`.

        Returns
        -------
        dict[str, list[str]]
            Mapping:
                collection_name -> list of saved crop paths
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: dict[str, list[str]] = {}

        for collection in self.collections:
            collection_name = collection.collection_name or "collection"
            collection_out_dir = out_dir / collection_name

            saved[collection_name] = collection.save_crops(
                out_dir=collection_out_dir,
                use_mask=use_mask,
                padding=padding,
                file_prefix=collection_name,
                ext=ext,
            )

        return saved
  
    def to_txt(
        self,
        out_dir: str | Path,
        *,
        include_conf: bool = True,
    ) -> None:
        """Write each collection as a YOLO segmentation .txt file under out_dir.

        Each file is named <collection_name>.txt.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for collection in self.collections:
            name = collection.collection_name or f"collection_{id(collection)}"
            collection.to_txt(out_dir / f"{name}.txt", include_conf=include_conf)

    def remap_classes(
        self,
        mapping: "ClassMapping | dict[int, int]",
        resolve: dict[str, str] | None = None,
    ) -> "Album":
        """Return a new Album with class_ids remapped across all collections.

        Parameters
        ----------
        mapping : ClassMapping (from build_class_mapping()) or a plain {old_id: new_id} dict.
        resolve : {old_label: chosen_new_label} — required for any ambiguous (list) entry.
            e.g. {'Daphnia': 'S_vetulus'}
        """
        return Album(
            collections=[c.remap_classes(mapping, resolve=resolve) for c in self.collections],
            album_name=self.album_name,
        )

    def class_counts(self) -> Counter:
        """Sum class_counts across every Collection in an Album."""
        total = Counter()
        for col in self.collections:
            total.update(col.class_counts)
        return total