"""The :class:`Album` container: a named set of collections."""

from __future__ import annotations

import os
import random
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from .collection import Collection
from .errors import HiresWarning

if TYPE_CHECKING:
    import pandas as pd

    from .classes import ClassMapping

__all__ = ["Album"]


class Album(Sequence[Collection]):
    """An immutable, ordered set of :class:`~hires.Collection` objects.

    An album typically holds the annotations of a whole dataset, one
    collection per image. Collections can be looked up by position or by
    name, and every transforming method returns a new album.

    Parameters
    ----------
    collections : iterable of Collection, optional
        The collections. Each must have a non-empty ``name``.
    name : str, optional
        Identifier of the album.

    Warns
    -----
    HiresWarning
        If several collections share a name. Later duplicates are renamed
        ``"{name}__1"``, ``"{name}__2"``, ... so that names stay unique.

    See Also
    --------
    hires.read_album : Load an album from a directory of label files.

    Examples
    --------
    >>> import hires
    >>> album = hires.Album(
    ...     [hires.Collection(name="a"), hires.Collection(name="b")], name="demo"
    ... )
    >>> album.names
    ['a', 'b']
    >>> album["b"]
    Collection(name='b', n=0)
    >>> "a" in album
    True
    """

    __slots__ = ("_by_name", "_collections", "_name")

    def __init__(
        self,
        collections: Iterable[Collection] = (),
        *,
        name: str | None = None,
    ) -> None:
        seen: Counter[str] = Counter()
        unique: list[Collection] = []
        renamed: list[tuple[str, str]] = []

        for collection in collections:
            if not isinstance(collection, Collection):
                raise TypeError(
                    f"Album items must be Collection objects, got {type(collection).__name__}."
                )
            if not collection.name:
                raise ValueError("Every collection in an Album needs a non-empty name.")
            original = collection.name
            if seen[original]:
                new_name = f"{original}__{seen[original]}"
                collection = collection.replace(name=new_name)
                renamed.append((original, new_name))
            seen[original] += 1
            unique.append(collection)

        if renamed:
            warnings.warn(
                f"{len(renamed)} collection(s) with duplicate names were renamed "
                f"(e.g. {renamed[0][0]!r} -> {renamed[0][1]!r}).",
                HiresWarning,
                stacklevel=2,
            )

        self._collections = tuple(unique)
        self._by_name = {c.name: c for c in self._collections}
        self._name = None if name is None else str(name)

    # ------------------------------------------------------------------ #
    # Metadata and lookup
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str | None:
        """Identifier of the album."""
        return self._name

    @property
    def collections(self) -> tuple[Collection, ...]:
        """The collections as a tuple."""
        return self._collections

    @property
    def names(self) -> list[str]:
        """Names of the collections, in order."""
        return [c.name for c in self._collections]  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self._collections)

    def __iter__(self) -> Iterator[Collection]:
        return iter(self._collections)

    @overload
    def __getitem__(self, key: int | str) -> Collection: ...
    @overload
    def __getitem__(self, key: slice) -> Album: ...
    def __getitem__(self, key: int | slice | str) -> Collection | Album:
        if isinstance(key, str):
            try:
                return self._by_name[key]
            except KeyError:
                raise KeyError(f"No collection named {key!r} in the album.") from None
        if isinstance(key, slice):
            return Album(self._collections[key], name=self._name)
        return self._collections[key]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self._by_name
        return item in self._collections

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Album):
            return NotImplemented
        return self._name == other._name and self._collections == other._collections

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        n_annotations = sum(len(c) for c in self._collections)
        return (
            f"Album(name={self._name!r}, collections={len(self)}, "
            f"annotations={n_annotations})"
        )

    def get(self, name: str, default: Any = None) -> Collection | Any:
        """Return the collection called ``name``, or ``default`` if absent."""
        return self._by_name.get(name, default)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dir(cls, directory: str | os.PathLike[str], **kwargs: Any) -> Album:
        """Load an album from a directory of YOLO label files.

        Alias of :func:`hires.read_album`; see there for the parameters.
        """
        from .io import read_album

        return read_album(directory, **kwargs)

    def map(self, func: Callable[[Collection], Collection]) -> Album:
        """Apply ``func`` to every collection and return the resulting album.

        Examples
        --------
        >>> confident = album.map(  # doctest: +SKIP
        ...     lambda c: c.filter(lambda a: a.confidence >= 0.8)
        ... )
        """
        return Album((func(c) for c in self._collections), name=self._name)

    def select(self, names: Iterable[str]) -> Album:
        """Return an album with only the collections called ``names``.

        Unknown names are ignored; the original order is kept.
        """
        wanted = set(names)
        return Album((c for c in self._collections if c.name in wanted), name=self._name)

    def filter_confidence(self, minimum: float) -> Album:
        """Keep the annotations with a confidence of at least ``minimum``.

        See :meth:`Collection.filter_confidence`.
        """
        return self.map(lambda c: c.filter_confidence(minimum))

    def filter_area(
        self, min_area: float | None = None, max_area: float | None = None
    ) -> Album:
        """Keep the annotations whose area lies in ``[min_area, max_area]``.

        See :meth:`Collection.filter_area`.
        """
        return self.map(lambda c: c.filter_area(min_area, max_area))

    def with_scale(self, dpi: float | None, unit: str | None) -> Album:
        """Return a copy in which every collection measures in physical units.

        See :meth:`Collection.with_scale`.
        """
        return self.map(lambda c: c.with_scale(dpi, unit))

    # ------------------------------------------------------------------ #
    # Class handling
    # ------------------------------------------------------------------ #
    def class_counts(self) -> dict[int, int]:
        """Count the annotations per class id across all collections."""
        counts: Counter[int] = Counter()
        for collection in self._collections:
            counts.update(collection.class_counts())
        return dict(sorted(counts.items()))

    def keep_classes(self, class_ids: int | Iterable[int]) -> Album:
        """Return an album with only annotations of the given class id(s)."""
        ids = {class_ids} if isinstance(class_ids, int) else set(class_ids)
        return self.map(lambda c: c.keep_classes(ids))

    def drop_classes(self, class_ids: int | Iterable[int]) -> Album:
        """Return an album without annotations of the given class id(s)."""
        ids = {class_ids} if isinstance(class_ids, int) else set(class_ids)
        return self.map(lambda c: c.drop_classes(ids))

    def remap_classes(
        self,
        mapping: ClassMapping | Mapping[int, int],
        resolve: Mapping[str, str] | None = None,
    ) -> Album:
        """Return a copy with class ids translated through ``mapping``.

        See :meth:`Collection.remap_classes`.
        """
        return self.map(lambda c: c.remap_classes(mapping, resolve))

    def balance(
        self,
        majority_classes: Iterable[int],
        *,
        max_ratio: float = 3.0,
        seed: int | None = 42,
    ) -> Album:
        """Drop majority-only collections to reduce class imbalance.

        A collection is *majority-only* when all of its annotations belong to
        ``majority_classes`` (empty collections count too). Such collections
        are removed in random order until::

            n_majority_annotations <= max_ratio * n_minority_annotations

        Collections containing any minority annotation are always kept.

        Parameters
        ----------
        majority_classes : iterable of int
            Over-represented class ids.
        max_ratio : float, default 3.0
            Largest accepted majority-to-minority annotation ratio.
        seed : int or None, default 42
            Seed for the random selection and the final shuffle.

        Returns
        -------
        Album
            The balanced album, in shuffled order.
        """
        majority = set(majority_classes)
        rng = random.Random(seed)

        def n_majority(collection: Collection) -> int:
            return sum(a.class_id in majority for a in collection)

        majority_only = [c for c in self._collections if n_majority(c) == len(c)]
        mixed = [c for c in self._collections if n_majority(c) != len(c)]

        n_minority = sum(len(c) - n_majority(c) for c in mixed)
        budget = int(max_ratio * n_minority)
        running = sum(n_majority(c) for c in mixed)

        rng.shuffle(majority_only)
        kept = []
        for collection in majority_only:
            count = n_majority(collection)
            if running + count <= budget:
                kept.append(collection)
                running += count

        balanced = mixed + kept
        rng.shuffle(balanced)
        return Album(balanced, name=self._name)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def summary(self) -> pd.DataFrame:
        """Return one row per collection with its size and image metadata."""
        import pandas as pd

        rows = [
            {
                "name": c.name,
                "n_annotations": len(c),
                "image_path": None if c.image_path is None else str(c.image_path),
                "image_width": None if c.image_size is None else c.image_size[0],
                "image_height": None if c.image_size is None else c.image_size[1],
                "dpi": c.dpi,
                "unit": c.unit,
            }
            for c in self._collections
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "name",
                "n_annotations",
                "image_path",
                "image_width",
                "image_height",
                "dpi",
                "unit",
            ],
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Return one flat dict per annotation across all collections.

        Same fields as :meth:`Collection.to_records` plus ``album``.
        """
        records = []
        missing = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", HiresWarning)
            for collection in self._collections:
                missing += collection.image_size is None and len(collection) > 0
                for row in collection.to_records():
                    records.append({"album": self._name, **row})
        if missing:
            warnings.warn(
                f"{missing} collection(s) have no image_size; their traits are "
                "measured in normalized coordinates. Load the album with "
                "image_dir=... to measure in pixels.",
                HiresWarning,
                stacklevel=2,
            )
        return records

    def to_frame(self) -> pd.DataFrame:
        """Return every annotation and its traits as a ``pandas.DataFrame``.

        Same columns as :meth:`Collection.to_frame`, preceded by ``album``.
        """
        import pandas as pd

        from .annotation import TRAIT_NAMES

        columns = [
            "album",
            "name",
            "image_path",
            "image_width",
            "image_height",
            "dpi",
            "unit",
            "class_id",
            "confidence",
            *TRAIT_NAMES,
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", HiresWarning)
            records = self.to_records()
        for warning in caught:
            warnings.warn(warning.message, warning.category, stacklevel=2)
        return pd.DataFrame(records, columns=columns)

    def to_csv(self, path: str | os.PathLike[str], **kwargs: Any) -> None:
        """Write :meth:`to_frame` to a CSV file.

        ``**kwargs`` are forwarded to :meth:`pandas.DataFrame.to_csv`;
        ``index`` defaults to ``False``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("index", False)
        self.to_frame().to_csv(path, **kwargs)

    def to_txt(self, directory: str | os.PathLike[str], *, confidence: bool = True) -> list[Path]:
        """Write every collection as ``{directory}/{name}.txt``.

        See :meth:`Collection.to_txt`.

        Returns
        -------
        list of pathlib.Path
            The written files.
        """
        directory = Path(directory)
        paths = []
        for collection in self._collections:
            path = directory / f"{collection.name}.txt"
            collection.to_txt(path, confidence=confidence)
            paths.append(path)
        return paths

    def save_crops(
        self,
        directory: str | os.PathLike[str],
        **kwargs: Any,
    ) -> dict[str, list[Path]]:
        """Save crops of every collection into ``{directory}/{name}/``.

        ``**kwargs`` are forwarded to :meth:`Collection.save_crops`.

        Returns
        -------
        dict of str to list of pathlib.Path
            The written files per collection name.
        """
        directory = Path(directory)
        return {
            c.name: c.save_crops(directory / c.name, **kwargs)  # type: ignore[misc]
            for c in self._collections
        }
