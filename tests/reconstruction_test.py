"""Tests for cubats.slide_collection.reconstruction.

Uses real (tiny) tiles on disk instead of mocking pyvips, so it catches problems with
8-bit/JPEG output, band handling, missing tiles and edge-tile cropping.
"""

# Standard Library
from types import SimpleNamespace

# Third Party
import numpy as np
import pytest
from PIL import Image

# CuBATS
from cubats.reconstruction import (assert_same_grid, save_thumbnail,
                                   stitch_tiles)

pyvips = pytest.importorskip("pyvips")

TILE = 8
BIG = 32  # tile size of the multi-tile test: large, so JPEG ringing at tile edges stays away from sampled pixels
RED, GREEN, BLUE = (220, 20, 20), (20, 220, 20), (20, 20, 220)
TOL = 20  # output is JPEG compressed, so allow deviations


def _save_tile(directory, col, row, size, color, mode="RGB"):
    """size is (width, height)."""
    Image.new(mode, size, color).save(directory / f"{col}_{row}.tif")


def _read(path):
    img = pyvips.Image.new_from_file(str(path))
    return np.ndarray(
        buffer=img.write_to_memory(),
        dtype=np.uint8,
        shape=[img.height, img.width, img.bands],
    )


def _close(pixel, expected):
    return np.allclose(pixel, expected, atol=TOL, rtol=0)


@pytest.fixture
def tile_dir(tmp_path):
    """2x2 grid, full size 50x45: narrow right-edge tiles, bottom-left tile missing."""
    d = tmp_path / "tiles"
    d.mkdir()
    _save_tile(d, 0, 0, (BIG, BIG), RED)
    _save_tile(d, 1, 0, (18, BIG), GREEN)
    _save_tile(d, 1, 1, (18, 13), BLUE)
    return d


def test_missing_and_edge_tiles(tile_dir, tmp_path):
    out = tmp_path / "out" / "wsi.tif"
    found = stitch_tiles(
        str(tile_dir), str(out), (2, 2), (50, 45), tile_size=BIG, progress=False
    )
    assert found == 3

    arr = _read(out)
    assert arr.shape == (45, 50, 3)
    assert _close(arr[16, 16], RED)
    assert _close(arr[16, 41], GREEN)
    assert _close(arr[38, 16], 192)
    assert _close(arr[38, 41], BLUE)


@pytest.mark.parametrize("mode, color", [("L", 100), ("RGBA", (100, 100, 100, 255))])
def test_grey_and_rgba_tiles_become_rgb(tmp_path, mode, color):
    d = tmp_path / "tiles"
    d.mkdir()
    _save_tile(d, 0, 0, (8, 8), color, mode)
    out = tmp_path / "wsi.tif"
    stitch_tiles(str(d), str(out), (1, 1), (8, 8), tile_size=TILE, progress=False)

    arr = _read(out)
    assert arr.shape == (8, 8, 3)
    assert _close(arr[4, 4], 100)


def test_single_channel_mask(tmp_path):
    d = tmp_path / "tiles"
    d.mkdir()
    _save_tile(d, 0, 0, (8, 8), 255, "L")  # (1, 0) missing
    out = tmp_path / "mask.tif"
    stitch_tiles(
        str(d), str(out), (2, 1), (16, 8),
        tile_size=TILE, bands=1, background=0, progress=False,
    )

    arr = _read(out)
    assert arr.shape == (8, 16, 1)
    assert _close(arr[4, 4, 0], 255)
    assert _close(arr[4, 12, 0], 0)


def test_thumbnail(tile_dir, tmp_path):
    thumb = tmp_path / "thumb.png"
    stitch_tiles(
        str(tile_dir), str(tmp_path / "wsi.tif"), (2, 2), (50, 45),
        tile_size=BIG, progress=False,
        thumbnail_path=str(thumb), thumbnail_size=4,
    )
    assert thumb.exists()
    assert max(Image.open(thumb).size) <= 4


def test_thumbnail_failure_is_not_raised(tmp_path):
    assert save_thumbnail(str(tmp_path / "nope.tif"), str(tmp_path / "t.png")) is False


def test_errors(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        stitch_tiles(str(tmp_path / "missing"), str(tmp_path / "o.tif"), (1, 1), (8, 8))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No tiles"):
        stitch_tiles(
            str(empty), str(tmp_path / "o.tif"), (1, 1), (8, 8),
            tile_size=TILE, progress=False,
        )


def _slide(name, grid, size):
    tiles = SimpleNamespace(level_tiles=[(1, 1), grid], level_dimensions=[(1, 1), size])
    return SimpleNamespace(name=name, tiles=tiles)


def test_assert_same_grid():
    a = _slide("a", (3, 2), (3000, 2000))
    b = _slide("b", (3, 2), (3000, 2000))
    assert assert_same_grid([a, b]) == ((3, 2), (3000, 2000))

    with pytest.raises(ValueError, match="mismatch"):
        assert_same_grid([a, _slide("c", (3, 2), (2999, 2000))])
    with pytest.raises(ValueError, match="mismatch"):
        assert_same_grid([a, _slide("d", (4, 2), (3000, 2000))])
