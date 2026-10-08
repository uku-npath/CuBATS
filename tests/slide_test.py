# Standard Library
import os
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import patch

# Third Party
import pytest
from PIL import Image as PILImage

# CuBATS
from cubats.reconstruction import DEFAULT_TILE_SIZE, get_deepzoom_grid
from cubats.slide_collection.slide import Slide


class TestSlideQuantify(unittest.TestCase):
    def setUp(self):
        # temporary dirs
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.dst = os.path.join(self.tmp.name, "dst")
        os.makedirs(self.src, exist_ok=True)
        os.makedirs(self.dst, exist_ok=True)

        # test fixture from repo tests/test_files
        self.test_file = os.path.join(
            os.path.dirname(__file__), "test_files", "test_file.tiff"
        )
        assert os.path.exists(
            self.test_file), "missing test fixture test_file.tiff"

        # copy fixture to src for clarity (Slide uses file path)
        self.slide_path = os.path.join(self.src, "Pat_Test.tiff")
        shutil.copy(self.test_file, self.slide_path)

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except Exception:
            pass

    def test_quantify_slide_raises_for_mask_and_reference(self):
        # mask slide should raise
        mask_slide = Slide("MaskSlide", self.slide_path, is_mask=True)
        with self.assertRaises(ValueError):
            mask_slide.quantify_slide([(0, 0)], save_dir=self.dst)

        # reference slide should raise
        ref_slide = Slide("RefSlide", self.slide_path, is_reference=True)
        with self.assertRaises(ValueError):
            ref_slide.quantify_slide([(0, 0)], save_dir=self.dst)

    def test_quantify_slide_requires_img_dir_when_save_img_true(self):
        s = Slide("S_requires_img_dir", self.slide_path)
        with self.assertRaises(ValueError):
            s.quantify_slide([(0, 0)], save_dir=self.dst,
                             save_img=True, img_dir=None)

    def test_quantify_slide_processes_and_saves_pickle_tile_level(self):
        s = Slide("S_process", self.slide_path)
        # Build a fake tile-result expected by summarize_quantification_results
        fake_tile_result = {
            "Flag": 1,
            "Zones": [100, 50, 25, 25, 0],  # 5 zones
            "Mask Count": 1024,  # > 0 to avoid division by zero
        }

        # Patch ProcessPoolExecutor so map returns our fake_tile_result iterator
        with patch("concurrent.futures.ProcessPoolExecutor") as MockExec:
            mock_executor = MockExec.return_value.__enter__.return_value
            mock_executor.map.return_value = iter([fake_tile_result])

            # run quantify_slide for one tile coordinate
            coords = [(0, 0)]
            s.quantify_slide(coords, save_dir=self.dst, save_img=False)

        # assert pickle file exists and contains the same dict
        out_pickle = os.path.join(self.dst, f"{s.name}_processing_info.pickle")
        self.assertTrue(os.path.exists(out_pickle))
        with open(out_pickle, "rb") as fh:
            loaded = pickle.load(fh)
        self.assertEqual(loaded, s.detailed_quantification_results)
        self.assertIn("Name", s.quantification_summary)
        self.assertEqual(s.quantification_summary["Name"], s.name)

    def test_quantify_slide_with_save_img_creates_img_dir(self):
        s = Slide("S_with_img", self.slide_path)
        fake_tile_result = {
            "Flag": 1,
            "Zones": [10, 10, 10, 10, 0],
            "Mask Count": 512,
        }
        img_dir = os.path.join(self.dst, "tiles_out")
        with patch("concurrent.futures.ProcessPoolExecutor") as MockExec:
            mock_executor = MockExec.return_value.__enter__.return_value
            mock_executor.map.return_value = iter([fake_tile_result])

            s.quantify_slide([(0, 0)], save_dir=self.dst,
                             save_img=True, img_dir=img_dir)

        # dab_tile_dir should be set and directory exists
        self.assertEqual(s.dab_tile_dir, img_dir)
        self.assertTrue(os.path.isdir(img_dir))
        # pickle exists as well
        out_pickle = os.path.join(self.dst, f"{s.name}_processing_info.pickle")
        self.assertTrue(os.path.exists(out_pickle))


class TestSlideReconstruct(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.out = os.path.join(self.tmp.name, "out")
        os.makedirs(self.src, exist_ok=True)
        os.makedirs(self.out, exist_ok=True)

        # small valid test fixture used elsewhere
        self.test_file = os.path.join(
            os.path.dirname(__file__), "test_files", "test_file.tiff"
        )
        assert os.path.exists(
            self.test_file), "missing test fixture test_file.tiff"

        # copy fixture and construct a real Slide (openslide will read the fixture)
        self.slide_path = os.path.join(self.src, "Pat_Test.tiff")
        shutil.copy(self.test_file, self.slide_path)
        self.slide = Slide("Pat_Test", self.slide_path)

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except Exception:
            pass

    def test_reconstruct_slide_raises_when_in_path_missing(self):
        missing = os.path.join(self.tmp.name, "no_such_dir")
        with self.assertRaises(ValueError):
            self.slide.reconstruct_slide(missing, self.out)

    def test_reconstruct_slide_raises_when_no_tif_files(self):
        empty_dir = os.path.join(self.tmp.name, "empty_tiles")
        os.makedirs(empty_dir, exist_ok=True)
        with self.assertRaises(ValueError):
            self.slide.reconstruct_slide(empty_dir, self.out)

    def test_reconstruct_slide_passes_grid_paths_and_thumbnail(self):
        with patch("cubats.slide_collection.slide.stitch_tiles", return_value=1) as m:
            self.slide.reconstruct_slide("in_dir", "out_dir", thumbnail=True)

        args, kwargs = m.call_args
        grid, size = get_deepzoom_grid(self.slide)
        name = self.slide.name
        self.assertEqual(args[0], "in_dir")
        self.assertEqual(args[1], os.path.join("out_dir", f"{name}_DAB.tif"))
        self.assertEqual(args[2:4], (grid, size))
        self.assertEqual(kwargs["tile_size"], DEFAULT_TILE_SIZE)
        self.assertEqual(
            kwargs["thumbnail_path"], os.path.join("out_dir", f"{name}_DAB_thumbnail.png"))

    def test_reconstruct_slide_defaults_and_no_thumbnail(self):
        self.slide.dab_tile_dir = "dab_dir"
        with patch("cubats.slide_collection.slide.stitch_tiles", return_value=1) as m:
            self.slide.reconstruct_slide(out_path="out_dir")
        self.assertEqual(m.call_args.args[0], "dab_dir")          # in_path defaults to dab_tile_dir
        self.assertIsNone(m.call_args.kwargs["thumbnail_path"])   # thumbnail is off by default

    def test_reconstruct_slide_requires_paths(self):
        self.slide.dab_tile_dir = None
        with self.assertRaises(ValueError):
            self.slide.reconstruct_slide(out_path="out_dir")      # no in_path and no dab_tile_dir
        with self.assertRaises(ValueError):
            self.slide.reconstruct_slide("in_dir")                # no out_path

    def test_reconstruct_slide_output_matches_deepzoom_size(self):
        pyvips = pytest.importorskip("pyvips")
        w, h = self.slide.tiles.level_dimensions[-1]
        tiles_dir = os.path.join(self.tmp.name, "tiles")
        os.makedirs(tiles_dir, exist_ok=True)
        PILImage.new("RGB", (min(1024, w), min(1024, h)), (10, 20, 30)).save(
            os.path.join(tiles_dir, "0_0.tif"))

        out_dir = os.path.join(self.out, "reconst_out")
        self.slide.reconstruct_slide(tiles_dir, out_dir)

        img = pyvips.Image.new_from_file(os.path.join(out_dir, f"{self.slide.name}_DAB.tif"))
        self.assertEqual((img.width, img.height), (w, h))


if __name__ == "__main__":
    unittest.main()
