# Tiling (`hires.tiling`)

Splitting images and their annotations into overlapping tiles, and merging
per-tile annotations back into full-image coordinates. A tile is identified
by the pixel coordinates `(x, y)` of its top-left corner; tiles on disk are
named `{stem}_{x}_{y}.png`. See
[Tiling training data](../user-guide/tiling.md) for a worked example.

| Function | Description |
|----------|-------------|
| [`tile_grid`][hires.tiling.tile_grid] | Tile positions for an image size |
| [`iter_tiles`][hires.tiling.iter_tiles] | Tiles of an image array, in memory |
| [`tile_image`][hires.tiling.tile_image] | Tiles of an image file, written as PNG |
| [`tile_annotations`][hires.tiling.tile_annotations] | Per-tile collections of an annotated image |
| [`merge_tiles`][hires.tiling.merge_tiles] | Per-tile annotations in full-image coordinates |
| [`parse_tile_name`][hires.tiling.parse_tile_name] | `(stem, x, y)` from a tile file name |

## Images

::: hires.tiling.tile_grid

::: hires.tiling.iter_tiles

::: hires.tiling.Tile

::: hires.tiling.tile_image

::: hires.tiling.parse_tile_name

## Annotations

::: hires.tiling.tile_annotations

::: hires.tiling.merge_tiles
