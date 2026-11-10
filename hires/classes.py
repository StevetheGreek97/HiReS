"""Translating class ids between two labelling schemas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["ClassMapping", "build_class_mapping"]


@dataclass(frozen=True)
class ClassMapping:
    """A reusable translation between two class-name schemas.

    Build one with :func:`build_class_mapping` and pass it to
    :meth:`Collection.remap_classes` or :meth:`Album.remap_classes`.

    Parameters
    ----------
    mapping : dict
        ``{old_id: new_id}`` for one-to-one renames, or
        ``{old_id: [candidate_new_ids]}`` for an old class that was split
        into several new classes (an *ambiguous* entry).
    old_names : dict of int to str
        ``{old_id: old_label}``.
    new_names : dict of int to str
        ``{new_id: new_label}``.
    """

    mapping: Mapping[int, int | Sequence[int]]
    old_names: Mapping[int, str]
    new_names: Mapping[int, str]

    def flatten(self, resolve: Mapping[str, str] | None = None) -> dict[int, int]:
        """Resolve ambiguous entries and return a flat ``{old_id: new_id}`` dict.

        Parameters
        ----------
        resolve : mapping of str to str, optional
            ``{old_label: chosen_new_label}``. Required for every ambiguous
            entry.

        Raises
        ------
        KeyError
            If an ambiguous label is missing from ``resolve`` or the chosen
            label does not exist in the new schema.
        ValueError
            If the chosen label is not one of the declared candidates.
        """
        resolve = resolve or {}
        new_ids = {label: id_ for id_, label in self.new_names.items()}
        flat: dict[int, int] = {}

        for old_id, target in self.mapping.items():
            if isinstance(target, int):
                flat[old_id] = target
                continue

            old_label = self.old_names[old_id]
            candidates = [self.new_names[t] for t in target]
            if old_label not in resolve:
                raise KeyError(
                    f"{old_label!r} is ambiguous {candidates}; choose one with resolve."
                )
            chosen = resolve[old_label]
            if chosen not in new_ids:
                raise KeyError(f"{chosen!r} is not a label of the new schema.")
            if new_ids[chosen] not in target:
                raise ValueError(
                    f"resolve[{old_label!r}] = {chosen!r} is not among the "
                    f"declared candidates {candidates}."
                )
            flat[old_id] = new_ids[chosen]

        return flat

    def __repr__(self) -> str:
        lines = []
        for old_id, target in self.mapping.items():
            old_label = self.old_names.get(old_id, str(old_id))
            if isinstance(target, int):
                new = f"{target} ({self.new_names.get(target, '?')!r})"
            else:
                new = "[" + ", ".join(
                    f"{t} ({self.new_names.get(t, '?')!r})" for t in target
                ) + "]"
            lines.append(f"  {old_id} ({old_label!r}) -> {new}")
        return "ClassMapping(\n" + "\n".join(lines) + "\n)"


def build_class_mapping(
    old_names: Mapping[int, str],
    new_names: Mapping[int, str],
    name_map: Mapping[str, str | Sequence[str]],
) -> ClassMapping:
    """Build a :class:`ClassMapping` between two class-name schemas.

    Parameters
    ----------
    old_names : mapping of int to str
        ``{old_id: old_label}`` of the schema the annotations currently use.
    new_names : mapping of int to str
        ``{new_id: new_label}`` of the target schema.
    name_map : mapping of str to (str or sequence of str)
        ``{old_label: new_label}`` for a direct rename, or
        ``{old_label: [candidate_new_labels]}`` for a class that is split in
        the new schema. Ambiguous entries are resolved per call to
        ``remap_classes``.

    Returns
    -------
    ClassMapping

    Raises
    ------
    KeyError
        If an old label has no entry in ``name_map`` or a target label does
        not exist in ``new_names``.

    Examples
    --------
    >>> mapping = build_class_mapping(
    ...     old_names={0: "ballooned", 1: "Daphnia"},
    ...     new_names={0: "d_pulex", 1: "d_galeata", 2: "ballooned"},
    ...     name_map={"ballooned": "ballooned", "Daphnia": ["d_pulex", "d_galeata"]},
    ... )
    >>> mapping.flatten({"Daphnia": "d_galeata"})
    {0: 2, 1: 1}
    """
    new_ids = {label: id_ for id_, label in new_names.items()}

    def lookup(label: str) -> int:
        if label not in new_ids:
            raise KeyError(f"Target label {label!r} not found in new_names.")
        return new_ids[label]

    mapping: dict[int, int | tuple[int, ...]] = {}
    for old_id, old_label in old_names.items():
        if old_label not in name_map:
            raise KeyError(f"name_map has no entry for old label {old_label!r}.")
        target = name_map[old_label]
        if isinstance(target, str):
            mapping[old_id] = lookup(target)
        else:
            mapping[old_id] = tuple(lookup(label) for label in target)

    return ClassMapping(mapping=mapping, old_names=dict(old_names), new_names=dict(new_names))
