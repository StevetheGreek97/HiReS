import pytest
from conftest import square

import hires
from hires import Album, Collection, HiresWarning


@pytest.fixture
def album():
    return Album(
        [
            Collection([square(0.1, 0.1, class_id=0), square(0.4, 0.4, class_id=1)], name="a",
                       image_size=(100, 100)),
            Collection([square(0.1, 0.1, class_id=0)], name="b", image_size=(100, 100)),
            Collection(name="c", image_size=(100, 100)),
        ],
        name="demo",
    )


def test_lookup(album):
    assert len(album) == 3
    assert album.names == ["a", "b", "c"]
    assert album["b"].name == "b" and album[0].name == "a"
    assert "a" in album and "zzz" not in album and album[0] in album
    assert album.get("zzz") is None
    assert isinstance(album[:2], Album) and album[:2].names == ["a", "b"]
    with pytest.raises(KeyError, match="zzz"):
        album["zzz"]


def test_requires_names():
    with pytest.raises(ValueError, match="name"):
        Album([Collection()])
    with pytest.raises(TypeError):
        Album(["x"])


def test_duplicate_names_are_renamed_without_mutating_input():
    first, second = Collection(name="a"), Collection(name="a")
    with pytest.warns(HiresWarning, match="duplicate"):
        album = Album([first, second])
    assert album.names == ["a", "a__1"]
    assert second.name == "a"


def test_class_operations(album):
    assert album.class_counts() == {0: 2, 1: 1}
    assert album.drop_classes(1).class_counts() == {0: 2}
    assert album.keep_classes([1]).class_counts() == {1: 1}
    assert album.remap_classes({0: 7}).class_counts() == {1: 1, 7: 2}
    assert album.class_counts() == {0: 2, 1: 1}
    assert album.drop_classes(1).name == "demo"


def test_remap_with_class_mapping(album):
    mapping = hires.build_class_mapping(
        {0: "daphnia", 1: "ballooned"},
        {0: "pulex", 1: "galeata", 2: "ballooned"},
        {"daphnia": ["pulex", "galeata"], "ballooned": "ballooned"},
    )
    assert album.remap_classes(mapping, {"daphnia": "galeata"}).class_counts() == {1: 2, 2: 1}
    with pytest.raises(KeyError, match="ambiguous"):
        album.remap_classes(mapping)
    with pytest.raises(ValueError, match="candidates"):
        album.remap_classes(mapping, {"daphnia": "ballooned"})


def test_select_and_map(album):
    assert album.select(["c", "a", "nope"]).names == ["a", "c"]
    assert album.map(lambda c: c.replace(name=c.name.upper())).names == ["A", "B", "C"]


def test_frames(album):
    frame = album.to_frame()
    assert len(frame) == 3
    assert frame["album"].unique().tolist() == ["demo"]
    assert frame["name"].tolist() == ["a", "a", "b"]
    summary = album.summary()
    assert summary["n_annotations"].tolist() == [2, 1, 0]
    scaled = album.with_scale(254, "mm").to_frame()
    assert scaled["area"].iloc[0] == pytest.approx(frame["area"].iloc[0] * 0.01)


def test_to_frame_warns_once_without_image_size():
    album = Album([Collection([square(0, 0)], name="a"), Collection([square(0, 0)], name="b")])
    with pytest.warns(HiresWarning, match="2 collection") as record:
        album.to_frame()
    assert len(record) == 1


def test_to_txt_and_read_album(album, tmp_path):
    paths = album.to_txt(tmp_path / "labels")
    assert [p.name for p in paths] == ["a.txt", "b.txt", "c.txt"]
    loaded = hires.read_album(tmp_path / "labels")
    assert loaded.names == ["a", "b", "c"]
    assert loaded.name == "labels"
    assert loaded.class_counts() == album.class_counts()
    assert Album.from_dir(tmp_path / "labels", name="x").name == "x"


def test_balance():
    majority_only = [Collection([square(0, 0, class_id=0)] * 10, name=f"m{i}") for i in range(10)]
    mixed = Collection([square(0, 0, class_id=0), square(0, 0, class_id=1)], name="mixed")
    album = Album([*majority_only, mixed])
    balanced = album.balance({0}, max_ratio=11)
    counts = balanced.class_counts()
    assert "mixed" in balanced
    assert counts[1] == 1 and counts[0] == 11  # 1 from mixed + one 10-annotation collection
    assert balanced.names == album.balance({0}, max_ratio=11).names  # deterministic
