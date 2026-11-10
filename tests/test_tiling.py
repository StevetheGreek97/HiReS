import cv2
import numpy as np
import pytest
from conftest import square

import hires
from hires.tiling import (
    iter_tiles,
    merge_tiles,
    parse_tile_name,
    tile_annotations,
    tile_grid,
    tile_image,
)


def test_tile_grid():
    assert tile_grid((2500, 900), tile_size=1024, overlap=150) == [(0, 0), (874, 0), (1748, 0)]
    assert tile_grid((100, 100), tile_size=1024) == [(0, 0)]
    assert tile_grid((1024, 1024), tile_size=1024) == [(0, 0)]
    assert tile_grid((300, 250), tile_size=(200, 100), overlap=0) == [
        (0, 0), (0, 100), (0, 200), (200, 0), (200, 100), (200, 200),
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("image_size", "tile_size", "overlap"),
    [((640, 520), 256, 96), ((2500, 900), 1024, 150), ((1000, 1000), (300, 200), 50)],
)
def test_tile_grid_covers_image(image_size, tile_size, overlap):
    tile_w, tile_h = (tile_size, tile_size) if isinstance(tile_size, int) else tile_size
    covered = np.zeros(image_size[::-1], dtype=bool)
    for x, y in tile_grid(image_size, tile_size, overlap):
        assert x < image_size[0] and y < image_size[1]
        covered[y : y + tile_h, x : x + tile_w] = True
    assert covered.all()


@pytest.mark.parametrize("overlap", [-1, 100, 150])
def test_tile_grid_rejects_bad_overlap(overlap):
    with pytest.raises(ValueError, match="overlap"):
        tile_grid((500, 500), tile_size=100, overlap=overlap)


def test_tile_grid_rejects_bad_sizes():
    with pytest.raises(ValueError, match="tile_size"):
        tile_grid((500, 500), tile_size=0)
    with pytest.raises(ValueError, match="image_size"):
        tile_grid((0, 500))


def test_iter_tiles_pads_border_tiles():
    image = np.arange(30 * 50, dtype=np.uint16).reshape(30, 50)
    tiles = list(iter_tiles(image, tile_size=32, overlap=8))
    assert [(t.x, t.y) for t in tiles] == [(0, 0), (24, 0)]
    for tile in tiles:
        assert tile.pixels.shape == (32, 32) and tile.pixels.dtype == np.uint16
    assert np.array_equal(tiles[0].pixels[:30], image[:, :32])
    assert np.array_equal(tiles[1].pixels[:30, :26], image[:, 24:])
    assert not tiles[1].pixels[30:].any() and not tiles[1].pixels[:, 26:].any()
    colour = np.ones((10, 10, 3), dtype=np.uint8)
    assert next(iter_tiles(colour, tile_size=(16, 12), overlap=0)).pixels.shape == (12, 16, 3)


def test_tile_image(tmp_path, image_path):
    written = tile_image(image_path, tmp_path / "tiles", tile_size=256, overlap=56)
    assert [p.name for p in written] == ["scan_0_0.png", "scan_200_0.png"]
    assert all(cv2.imread(str(p)).shape == (256, 256, 3) for p in written)
    assert [parse_tile_name(p) for p in written] == [("scan", 0, 0), ("scan", 200, 0)]


def test_tile_image_keeps_bit_depth(tmp_path):
    source = tmp_path / "gray.tif"
    cv2.imwrite(str(source), np.full((40, 40), 40_000, dtype=np.uint16))
    (tile,) = tile_image(source, tmp_path / "out", tile_size=64, overlap=0)
    pixels = cv2.imread(str(tile), cv2.IMREAD_UNCHANGED)
    assert pixels.dtype == np.uint16 and pixels.shape == (64, 64) and pixels[0, 0] == 40_000


def test_tile_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        tile_image(tmp_path / "nope.png", tmp_path / "out")
    (tmp_path / "broken.png").write_text("not an image")
    with pytest.raises(OSError, match="decode"):
        tile_image(tmp_path / "broken.png", tmp_path / "out")


def test_parse_tile_name():
    assert parse_tile_name("scan_01_874_1748.png") == ("scan_01", 874, 1748)
    assert parse_tile_name("dir/a_b_0_0.txt") == ("a_b", 0, 0)
    for bad in ("scan.png", "scan_12.png", "scan_a_b.png"):
        with pytest.raises(ValueError, match="not a tile name"):
            parse_tile_name(bad)


def test_tile_annotations():
    collection = hires.Collection(
        [square(0.05, 0.05, 0.1), square(0.45, 0.4, 0.1, class_id=1), square(0.8, 0.8, 0.1)],
        name="scan",
        image_size=(1000, 1000),
    )
    tiles = tile_annotations(collection, tile_size=600, overlap=200)
    assert list(tiles) == [(0, 0), (0, 400), (400, 0), (400, 400)]
    assert [len(t) for t in tiles.values()] == [2, 1, 1, 2]
    assert tiles[(400, 0)].name == "scan_400_0" and tiles[(400, 0)].image_size == (600, 600)
    # The middle square straddles the border of tile (400, 0) and is not clipped.
    straddling = tiles[(400, 0)][0]
    assert straddling.class_id == 1
    assert straddling.bounds == pytest.approx((50 / 600, 400 / 600, 150 / 600, 500 / 600))
    assert tiles[(0, 0)][0].bounds == pytest.approx((50 / 600, 50 / 600, 150 / 600, 150 / 600))


def test_tile_annotations_needs_image_size():
    with pytest.raises(ValueError, match="image_size"):
        tile_annotations(hires.Collection([square(0.1, 0.1)]))
    tiles = tile_annotations(hires.Collection([square(0.1, 0.1)]), image_size=(100, 100))
    assert list(tiles) == [(0, 0)] and tiles[(0, 0)].name is None


def test_merge_tiles_inverts_tile_annotations():
    collection = hires.Collection(
        [square(0.05, 0.05, 0.1, confidence=0.8), square(0.45, 0.4, 0.1, class_id=1)],
        name="scan",
        image_size=(1000, 800),
    )
    tiles = tile_annotations(collection, tile_size=(600, 500), overlap=200)
    merged = merge_tiles(tiles, tile_size=(600, 500), image_size=(1000, 800), name="scan")
    assert merged.name == "scan" and merged.image_size == (1000, 800)
    assert len(merged) == sum(len(t) for t in tiles.values()) > len(collection)
    for restored in merged:
        original = collection[0] if restored.class_id == 0 else collection[1]
        assert restored.confidence == original.confidence
        assert restored.polygon.symmetric_difference(original.polygon).area < 1e-12
    # Duplicates from overlapping tiles collapse under NMS.
    scored = merged.replace(annotations=[a.replace(confidence=0.5) for a in merged])
    assert len(scored.nms(0.5)) == 2
