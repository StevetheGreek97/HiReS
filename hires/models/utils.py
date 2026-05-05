from __future__ import annotations

import dataclasses
from shapely.geometry import Polygon


@dataclasses.dataclass
class ClassMapping:
    """Reusable schema mapping produced by build_class_mapping().

    Carries old/new label context so remap_classes() can accept label names
    in resolve instead of raw integer IDs.

    Attributes
    ----------
    mapping   : {old_id: new_id} for direct renames,
                {old_id: [candidate_new_ids]} for ambiguous split classes.
    old_names : {old_id: old_label}
    new_names : {new_id: new_label}
    """

    mapping: dict[int, int | list[int]]
    old_names: dict[int, str]
    new_names: dict[int, str]

    def flatten(self, resolve: dict[str, str]) -> dict[int, int]:
        """Resolve ambiguous entries and return a flat {old_id: new_id} mapping.

        Parameters
        ----------
        resolve : {old_label: chosen_new_label}
            Required for every list entry in mapping.
            e.g. {'Daphnia': 'S_vetulus'}
        """
        inv_new = {label: id_ for id_, label in self.new_names.items()}
        result: dict[int, int] = {}

        for old_id, target in self.mapping.items():
            if isinstance(target, list):
                old_label = self.old_names[old_id]
                if old_label not in resolve:
                    candidates = [self.new_names[t] for t in target]
                    raise KeyError(
                        f"'{old_label}' is ambiguous {candidates}; add it to resolve."
                    )
                chosen_label = resolve[old_label]
                chosen_id = inv_new.get(chosen_label)
                if chosen_id is None:
                    raise KeyError(f"'{chosen_label}' not found in new_names.")
                if chosen_id not in target:
                    candidates = [self.new_names[t] for t in target]
                    raise ValueError(
                        f"resolve['{old_label}'] = '{chosen_label}' is not among "
                        f"the declared candidates {candidates}."
                    )
                result[old_id] = chosen_id
            else:
                result[old_id] = target

        return result

    def __repr__(self) -> str:
        lines = []
        for old_id, target in self.mapping.items():
            old_label = self.old_names.get(old_id, str(old_id))
            if isinstance(target, list):
                candidates = [f"{t} ({self.new_names.get(t, '?')!r})" for t in target]
                lines.append(f"  {old_id} ({old_label!r}) → [{', '.join(candidates)}]")
            else:
                new_label = self.new_names.get(target, str(target))
                lines.append(f"  {old_id} ({old_label!r}) → {target} ({new_label!r})")
        return "ClassMapping(\n" + "\n".join(lines) + "\n)"


def build_class_mapping(
    old_names: dict[int, str],
    new_names: dict[int, str],
    name_map: dict[str, str | list[str]],
) -> ClassMapping:
    """Build a reusable ClassMapping across two class-name schemas.

    Parameters
    ----------
    old_names : {old_id: old_label}
        e.g. {0: 'ballooned', 1: 'Daphnia'}
    new_names : {new_id: new_label}
        e.g. {0: 'd_pulex', 1: 'd_galeata', 2: 'S_vetulus', 3: 'ballooned'}
    name_map : {old_label: new_label | [candidate_new_labels]}
        str   → direct 1-to-1 rename.
        list  → ambiguous split class; resolve when calling remap_classes().
        e.g. {'Daphnia': ['S_vetulus', 'd_pulex', 'd_galeata'], 'ballooned': 'ballooned'}

    Returns
    -------
    ClassMapping
        Pass to Collection.remap_classes() or Album.remap_classes() with a
        label-based resolve dict to apply per-album species assignment.

    Raises
    ------
    KeyError  if a label is missing from name_map or not found in new_names.
    """
    inv_new = {label: id_ for id_, label in new_names.items()}
    mapping: dict[int, int | list[int]] = {}

    for old_id, old_label in old_names.items():
        target = name_map.get(old_label)
        if target is None:
            raise KeyError(f"name_map has no entry for old label '{old_label}'")

        if isinstance(target, list):
            resolved_ids: list[int] = []
            for t in target:
                if t not in inv_new:
                    raise KeyError(f"Candidate label '{t}' not found in new_names.")
                resolved_ids.append(inv_new[t])
            mapping[old_id] = resolved_ids
        else:
            if target not in inv_new:
                raise KeyError(f"Target label '{target}' not found in new_names.")
            mapping[old_id] = inv_new[target]

    return ClassMapping(mapping=mapping, old_names=old_names, new_names=new_names)


def _square_plot_span(polygon: Polygon, padding: float) -> tuple[float, float, float]:
    minx, miny, maxx, maxy = polygon.bounds
    dx, dy = maxx - minx, maxy - miny
    cx, cy = (maxx + minx) / 2, (maxy + miny) / 2
    max_dim = max(dx, dy)
    span = (max_dim / 2) * (1 + padding * 2)
    return cx, cy, span

