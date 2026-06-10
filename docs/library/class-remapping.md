# Class remapping

`hires.models.build_class_mapping` · `hires.models.ClassMapping`

[`Collection.remap_classes`](collection.md#remap_classesmapping-resolvenone) and
[`Album.remap_classes`](album.md#remap_classesmapping-resolvenone) rewrite the
integer `class_id` of every annotation. There are two ways to describe the remap:

1. **A plain `{old_id: new_id}` dict** — direct integer→integer renames. Any
   `class_id` not present in the dict is left unchanged.
2. **A `ClassMapping`** built with `build_class_mapping()` — a reusable,
   *label-aware* schema translation between two class-name dictionaries. This is
   the robust option when the source and target models use different class names
   and/or a different number of classes.

---

## `build_class_mapping(old_names, new_names, name_map)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `old_names` | `dict[int, str]` | `{id: label}` of the **source** schema (the IDs currently in your annotations) |
| `new_names` | `dict[int, str]` | `{id: label}` of the **target** schema (the IDs you want to end up with) |
| `name_map` | `dict[str, str \| list[str]]` | `{old_label: new_label}` for a direct rename, or `{old_label: [candidate_new_labels]}` when one old class can map to several new ones |

Returns a `ClassMapping`. It resolves each old label to a new **id** by looking
the chosen label up in `new_names`, so the two schemas can have completely
different ID orderings.

A list value marks an **ambiguous** entry: a single old class that could become
one of several new classes. You decide which one later, per-collection or
per-album, via the `resolve` argument of `remap_classes`. The chosen label must
be one of the declared candidates — picking anything else raises a `ValueError`.

---

## `ClassMapping`

| Attribute / Method | Description |
|--------------------|-------------|
| `mapping` | `{old_id: new_id}` for direct renames, `{old_id: [candidate_new_ids]}` for ambiguous ones |
| `old_names` / `new_names` | The two schemas it was built from |
| `flatten(resolve)` | Resolve all ambiguous entries to a flat `{old_id: new_id}` dict. `resolve` is `{old_label: chosen_new_label}` |

`ClassMapping` has a readable `repr`, which is handy for inspecting a mapping
before applying it.

---

## Example 1 — merge + resolve an ambiguous class

A generic detector with two classes (`ballooned`, `Daphnia`) is translated into
a finer-grained species schema. `ballooned` maps 1:1, but `Daphnia` is ambiguous
— it could be any of three species — so it is declared as a list and resolved
when the mapping is applied.

```python
from hires.models import build_class_mapping, Collection

class_names_old = {0: "ballooned", 1: "Daphnia"}
class_names_new = {0: "d_pulex", 1: "d_galeata", 2: "S_vetulus", 3: "ballooned"}

SCHEMA = {
    "ballooned": "ballooned",                          # 1:1 rename
    "Daphnia":   ["S_vetulus", "d_pulex", "d_galeata"],  # ambiguous → resolve later
}

full_mapping = build_class_mapping(class_names_old, class_names_new, SCHEMA)

print(full_mapping)
# ClassMapping(
#   0 ('ballooned') → 3 ('ballooned')
#   1 ('Daphnia') → [2 ('S_vetulus'), 0 ('d_pulex'), 1 ('d_galeata')]
# )

# The SAME mapping is reused across collections, resolving the ambiguous
# 'Daphnia' class to a different species for each sample you know the identity of:
s_vet     = Collection.read_txt("samples/s_vet.txt")
d_pulex   = Collection.read_txt("samples/d_pulex.txt")
d_galeata = Collection.read_txt("samples/d_galeata.txt")

s_vet_remapped     = s_vet.remap_classes(full_mapping,     resolve={"Daphnia": "S_vetulus"})
d_pulex_remapped   = d_pulex.remap_classes(full_mapping,   resolve={"Daphnia": "d_pulex"})
d_galeata_remapped = d_galeata.remap_classes(full_mapping, resolve={"Daphnia": "d_galeata"})
# In every case  class_id 0 (ballooned) → 3;
# class_id 1 (Daphnia) → 2 (S_vetulus) / 0 (d_pulex) / 1 (d_galeata) respectively.
```

Build the ambiguous mapping once, then resolve it per collection (or per album)
— there is no need to rebuild it for each species.

Omitting `resolve` for an ambiguous entry raises a `KeyError` that lists the
candidate labels, so you can never silently mis-assign a split class.

---

## Example 2 — per-sample relabelling across an Album

When you know the species of each sample up front, you can collapse several
fine-grained labels into the correct one per sample using a same-schema map
(`build_class_mapping(names, names, schema)`). Because every entry is a 1:1
string rename, no `resolve` is needed.

```python
from pathlib import Path
from hires.models import build_class_mapping, Album

# One shared label↔id schema for both sides of the remap.
names = {
    0: "Dg_f_lateral_adult",
    1: "Dp_f_lateral_adult",
    2: "Sv_f_lateral_adult",
    3: "Daphnia_f_lateral_juvenile",
    4: "Sv_f_lateral_juvenile",
    5: "chydoride",
    6: "copepod",
    7: "unidentified_Daphniidae",
}

# Each sample's adults/juveniles are forced to the correct species.
S_VET_SCHEMA = {
    "Dg_f_lateral_adult":         "Sv_f_lateral_adult",
    "Dp_f_lateral_adult":         "Sv_f_lateral_adult",
    "Sv_f_lateral_adult":         "Sv_f_lateral_adult",
    "Daphnia_f_lateral_juvenile": "Sv_f_lateral_juvenile",
    "Sv_f_lateral_juvenile":      "Sv_f_lateral_juvenile",
    "chydoride":                  "chydoride",
    "copepod":                    "copepod",
    "unidentified_Daphniidae":    "unidentified_Daphniidae",
}
D_GAL_SCHEMA = {
    "Dg_f_lateral_adult":         "Dg_f_lateral_adult",
    "Dp_f_lateral_adult":         "Dg_f_lateral_adult",
    "Sv_f_lateral_adult":         "Dg_f_lateral_adult",
    "Daphnia_f_lateral_juvenile": "Daphnia_f_lateral_juvenile",
    "Sv_f_lateral_juvenile":      "Daphnia_f_lateral_juvenile",
    "chydoride":                  "chydoride",
    "copepod":                    "copepod",
    "unidentified_Daphniidae":    "unidentified_Daphniidae",
}
D_PUL_SCHEMA = {
    "Dg_f_lateral_adult":         "Dp_f_lateral_adult",
    "Dp_f_lateral_adult":         "Dp_f_lateral_adult",
    "Sv_f_lateral_adult":         "Dp_f_lateral_adult",
    "Daphnia_f_lateral_juvenile": "Daphnia_f_lateral_juvenile",
    "Sv_f_lateral_juvenile":      "Daphnia_f_lateral_juvenile",
    "chydoride":                  "chydoride",
    "copepod":                    "copepod",
    "unidentified_Daphniidae":    "unidentified_Daphniidae",
}

base_path = Path("data")

# s_vet, d_gal, d_pul are lists of .txt paths for each species' samples.
for schema, species_paths, name in zip(
    [S_VET_SCHEMA, D_GAL_SCHEMA, D_PUL_SCHEMA],
    [s_vet, d_gal, d_pul],
    ["s_vet", "d_gal", "d_pul"],
):
    full_mapping = build_class_mapping(names, names, schema)

    sp_album = Album.from_paths(species_paths, album_name=name)
    print(sp_album.class_counts())            # before remap

    sp_album_remapped = sp_album.remap_classes(full_mapping)
    print(sp_album_remapped.class_counts())   # after remap

    sp_album_remapped.to_txt(out_dir=base_path / "comadapt_model_E06_remaped")
```

`Album.remap_classes` returns a **new** `Album` (the original is untouched) and
applies the same mapping to every collection. `to_txt` then writes one
`<collection_name>.txt` per sample under `out_dir`.

!!! tip "When to use a plain dict vs. `build_class_mapping`"
    Reach for a plain `{old_id: new_id}` dict for quick, in-schema merges
    (`album.remap_classes({0: 0, 1: 0})`). Use `build_class_mapping` when you are
    translating between two named schemas — it validates every label against
    `new_names` and turns split classes into explicit, resolvable choices.
