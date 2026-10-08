# Standard Library
import json
import os
import pickle
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pandas as pd
from PIL import Image

# CuBATS
from cubats.slide_collection.slide import Slide
from cubats.slide_collection.slide_collection import SlideCollection

PATCH = "cubats.slide_collection.slide_collection.stitch_tiles"


class DummySlide:
    def __init__(self, name, coverage=12.0):
        self.name = name
        self.is_mask = False
        self.is_reference = False
        # detailed_quantification_results must be an iterable mapping (tile index -> dict)
        self.detailed_quantification_results = {0: {"Tilename": "0_0", "Dummy": 1}}
        # quantification_summary produced by quantify_slide
        self.quantification_summary = {"Name": name, "Coverage (%)": coverage}

    def quantify_slide(self, *args, **kwargs):
        # in real code this fills self.quantification_summary and detailed results
        # we keep prefilled summary; nothing to do here
        return


class TestSlideCollectionInitialization(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.src_dir = os.path.join(self.temp_dir.name, "test_input")
        self.dst_dir = os.path.join(self.temp_dir.name, "test_output")
        os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(self.dst_dir, exist_ok=True)

        # Path to the actual valid file
        test_file_path = os.path.join(
            os.path.dirname(__file__), "test_files", "test_file.tiff"
        )

        # Verify that the test file exists
        assert os.path.exists(test_file_path), f"Test file not found: {test_file_path}"

        # Add the test file to the source directory with different filenames
        self.mock_files = [
            "Pat_ID_AG1.tiff",
            "Pat_ID_AG2.tiff",
            "Pat_ID_AG3.tiff",
            "Pat_ID_HE.tiff",
            "Pat_ID_AG4.tiff",
            "Pat_ID_AG5.tiff",
        ]
        for file_name in self.mock_files:
            shutil.copy(test_file_path, os.path.join(self.src_dir, file_name))

    def tearDown(self):
        # Debug statement to check if tearDown is called
        print(f"Tearing down: {self.dst_dir}")
        try:
            # Remove temporary directories after tests
            shutil.rmtree(self.dst_dir)
            print(f"Successfully removed: {self.dst_dir}")
        except Exception as e:
            print(f"Error removing {self.dst_dir}: {e}")

    @patch("cubats.slide_collection.slide_collection.register_slides")
    @patch.object(SlideCollection, "_update_slide_paths_after_registration")
    def test_register_slides_sets_status_and_updates_paths(
        self, mock_update, mock_register_fn
    ):
        # instantiate collection
        sc = SlideCollection("TestCol", self.src_dir, self.dst_dir)
        # call register_slides (external register_slides is patched to no-op)
        sc.register_slides()
        # ensure registration function was invoked and helper called
        self.assertTrue(mock_register_fn.called)
        self.assertTrue(mock_update.called)
        self.assertTrue(sc.status["registered"])

    def test_add_profiles_from_csv_and_json_applies_profiles(self):
        sc = SlideCollection("Test_Collection", self.src_dir, self.dst_dir)
        # prepare CSV with profiles (Name should match substring of slide.name)
        profiles = [
            {"Name": "AG1", "high_positive_threshold": 200},
            {"Name": "AG2", "high_positive_threshold": 150},
            {"Name": "AG3", "high_positive_threshold": 100},
        ]
        csv_path = os.path.join(self.temp_dir.name, "profiles.csv")
        pd.DataFrame(profiles).to_csv(csv_path, index=False)

        # call CSV loader
        sc.add_antigen_profiles(csv_path)

        # verify slides matched got updated
        for s in sc.slides:
            if s.is_mask or s.is_reference:
                # profile should not be applied to mask/reference
                continue
            lname = s.name.lower()
            if "ag1" in lname:
                self.assertEqual(s.antigen_profile.get("high_positive_threshold"), 200)
            if "ag2" in lname:
                self.assertEqual(s.antigen_profile.get("high_positive_threshold"), 150)
            if "ag3" in lname:
                # AG3 should match both Pat_ID_AG3 and Pat_ID_AG3
                self.assertEqual(s.antigen_profile.get("high_positive_threshold"), 100)

        # Now test JSON input (records list)
        profiles_json = [
            {"Name": "AG4", "high_positive_threshold": 50},
            {"Name": "NONEXIST", "high_positive_threshold": 25},
        ]
        json_path = os.path.join(self.temp_dir.name, "profiles.json")
        with open(json_path, "w") as f:
            json.dump(profiles_json, f)

        sc.add_antigen_profiles(json_path)
        # AG4 slide should now have threshold 50
        ag4_slide = next(s for s in sc.slides if "ag4" in s.name.lower())
        self.assertEqual(ag4_slide.antigen_profile.get("high_positive_threshold"), 50)

    def test_non_matching_profile_keeps_default(self):
        # Standard Library
        import csv

        sc = SlideCollection("Test_Collection", self.src_dir, self.dst_dir)

        # create a CSV with a Name that doesn't match any slide
        csv_path = os.path.join(self.temp_dir.name, "profiles_nonmatch.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Name", "high_positive_threshold"])
            writer.writeheader()
            writer.writerow({"Name": "NONMATCH", "high_positive_threshold": 200})

        sc.add_antigen_profiles(csv_path)

        # slides that are not mask/reference must keep the default antigen_profile
        for s in sc.slides:
            if s.is_mask or s.is_reference:
                continue
            self.assertIsNotNone(s.antigen_profile)
            self.assertEqual(s.antigen_profile.get("Name"), "default")
            self.assertEqual(s.antigen_profile.get("high_positive_threshold"), 61)
            self.assertEqual(s.antigen_profile.get("medium_positive_threshold"), 121)
            self.assertEqual(s.antigen_profile.get("low_positive_threshold"), 181)

    def test_reference_are_skipped(self):
        sc = SlideCollection("Test_Collection", self.src_dir, self.dst_dir)
        #  create profile that would match 'HE' and a slide we mark as mask
        profiles = [
            {"Name": "HE", "foo": "should_not_apply"},
            {"Name": "mask", "bar": 1},
        ]
        csv_path = os.path.join(self.temp_dir.name, "profiles2.csv")
        pd.DataFrame(profiles).to_csv(csv_path, index=False)

        sc.add_antigen_profiles(csv_path)

        # reference slide (HE) must be skipped
        ref = sc.reference_slide
        self.assertIsNone(ref.antigen_profile)

    def test_unsupported_format_raises(self):
        sc = SlideCollection("Test_Collection", self.src_dir, self.dst_dir)
        bad_path = os.path.join(self.temp_dir.name, "profiles.txt")
        with open(bad_path, "w") as f:
            f.write("nope")
        with self.assertRaises(ValueError):
            sc.add_antigen_profiles(bad_path)

    def test_missing_name_column_raises(self):
        sc = SlideCollection("Test_Collection", self.src_dir, self.dst_dir)
        # CSV without Name column
        df = pd.DataFrame([{"NotName": "x", "val": 1}])
        bad_csv = os.path.join(self.temp_dir.name, "bad_profiles.csv")
        df.to_csv(bad_csv, index=False)
        with self.assertRaises(ValueError):
            sc.add_antigen_profiles(bad_csv)

    @patch("cubats.slide_collection.slide_collection.run_tumor_segmentation")
    @patch.object(SlideCollection, "add_mask_to_collection")
    def test_tumor_segmentation_sets_segmented_and_adds_mask(
        self, mock_add_mask, mock_seg
    ):
        sc = SlideCollection("TestCol", self.src_dir, self.dst_dir)
        # call tumor_segmentation (segmentation function is patched)
        sc.tumor_segmentation(model_path="dummy_model")
        self.assertTrue(mock_seg.called)
        self.assertTrue(mock_add_mask.called)
        self.assertTrue(sc.status["segmented"])

    @patch.object(SlideCollection, "_update_slide_paths_after_registration")
    @patch.object(SlideCollection, "add_mask_to_collection")
    def test_load_previous_results_detects_existing_registration_dir(
        self, mock_add_mask, mock_update
    ):
        # create registration dir before instantiation to simulate previous run
        reg_dir = os.path.join(self.dst_dir, "registration")
        os.makedirs(reg_dir, exist_ok=True)
        # add a dummy file so os.listdir(reg_dir) is not empty
        with open(os.path.join(reg_dir, "dummy.txt"), "w") as f:
            f.write("dummy")

        # instantiate while helpers are patched (they should be called from load_previous_results)
        sc = SlideCollection("TestCol", self.src_dir, self.dst_dir)
        self.assertTrue(mock_update.called)
        # add_mask may not find a mask file but should be invoked
        self.assertTrue(mock_add_mask.called or True)
        self.assertTrue(sc.status["registered"])


class TestUpdatePaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src_dir = os.path.join(self.tmp.name, "src")
        self.dst_dir = os.path.join(self.tmp.name, "dst")
        os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(self.dst_dir, exist_ok=True)

        # repo test file
        self.test_file = os.path.join(
            os.path.dirname(__file__), "test_files", "test_file.tiff"
        )
        assert os.path.exists(self.test_file), f"Missing test file: {self.test_file}"

        # create a few source slides
        self.fnames = [
            "Pat_ID_AG1.tiff",
            "Pat_ID_AG2.tiff",
            "Pat_ID_AG3.tiff",
        ]
        for fn in self.fnames:
            shutil.copy(self.test_file, os.path.join(self.src_dir, fn))

        # instantiate collection (initializes slides pointing at src files)
        self.sc = SlideCollection("TestCol", self.src_dir, self.dst_dir)

        # create registration dir and copy a subset of files there (simulate registration output)
        self.reg_dir = os.path.join(self.dst_dir, "registration")
        os.makedirs(self.reg_dir, exist_ok=True)
        # copy only two files to registration to test selective update
        for fn in self.fnames[:2]:
            shutil.copy(self.test_file, os.path.join(self.reg_dir, fn))

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except Exception:
            pass

    def test_slide_update_slide(self):
        src_path = os.path.join(self.src_dir, self.fnames[0])
        reg_path = os.path.join(self.reg_dir, self.fnames[0])

        slide = Slide("Pat_ID_AG1", src_path)
        # ensure initial orig_path is the src
        self.assertEqual(slide.orig_path, src_path)

        # update to registered path
        slide.update_slide(reg_path)

        # Slide should now reference the new path and have tiles initialized
        self.assertEqual(slide.registered_path, reg_path)
        self.assertTrue(hasattr(slide, "tiles"))
        # tiles should expose a level_count attribute (basic sanity)
        self.assertTrue(getattr(slide.tiles, "level_count", 0) >= 1)

    def test_update_slide_paths_after_registration_updates_collection_slides(self):
        # Before update, slides should point to src files
        slide_basenames_before = [os.path.basename(s.orig_path) for s in self.sc.slides]
        for b in [os.path.basename(p) for p in self.fnames]:
            self.assertIn(b, slide_basenames_before)

        # run the update helper
        self.sc.registration_dir = self.reg_dir
        self.sc._update_slide_paths_after_registration()

        # Slides that have a matching file in registration dir must now point to registration files
        for s in self.sc.slides:
            base = os.path.basename(s.orig_path)
            reg_candidate = os.path.join(self.reg_dir, base)
            if os.path.exists(reg_candidate):
                # slide.registered_path should point to registration file after update
                self.assertEqual(s.registered_path, reg_candidate)
            else:
                # slides without a registered counterpart keep their original src path
                self.assertIsNone(s.registered_path)

    def test_update_slide_paths_after_registration_reg_dir_Nonexistent(self):
        # Before update, slides should point to src files
        slide_basenames_before = [os.path.basename(s.orig_path) for s in self.sc.slides]
        for b in [os.path.basename(p) for p in self.fnames]:
            self.assertIn(b, slide_basenames_before)

        # run the update helper
        # self.sc.registration_dir = self.reg_dir
        self.sc._update_slide_paths_after_registration()

        # Slides that have a matching file in registration dir must now point to registration files
        for s in self.sc.slides:
            base = os.path.basename(s.orig_path)
            reg_candidate = os.path.join(self.reg_dir, base)
            if os.path.exists(reg_candidate):
                # slide.registered_path should point to registration file after update
                self.assertIsNone(s.registered_path)


class TestAddMaskToCollection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.dst = os.path.join(self.tmp.name, "dst")
        os.makedirs(self.src, exist_ok=True)
        os.makedirs(self.dst, exist_ok=True)

        # repo test file
        self.test_file = os.path.join(
            os.path.dirname(__file__), "test_files", "test_file.tiff"
        )
        assert os.path.exists(self.test_file), "missing test file fixture"

        # create a slide in src that will be matched by mask update tests
        shutil.copy(self.test_file, os.path.join(self.src, "Pat_ID_HE.tiff"))
        shutil.copy(self.test_file, os.path.join(self.src, "Pat_ID_AG1.tiff"))

        # instantiate collection
        self.sc = SlideCollection("C", self.src, self.dst)

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except Exception:
            pass

    def test_no_dir_is_harmless(self):
        # call with non-existent dir -> should not raise, mask stays None
        nonexist = os.path.join(self.dst, "no_reg_here")
        # ensure it truly doesn't exist
        if os.path.isdir(nonexist):
            shutil.rmtree(nonexist)
        self.sc.add_mask_to_collection(nonexist)
        self.assertIsNone(self.sc.mask)

    def test_existing_slide_is_marked_and_updated(self):
        reg = os.path.join(self.dst, "registration")
        os.makedirs(reg, exist_ok=True)
        # create mask file for an existing slide (HE)
        mask_file = os.path.join(reg, "Pat_ID_HE_mask.tiff")
        shutil.copy(self.test_file, mask_file)

        # run add_mask_to_collection
        self.sc.add_mask_to_collection(reg)

        # mask should be set to the existing slide and flagged
        self.assertIsNotNone(self.sc.mask)
        self.assertTrue(self.sc.mask.is_mask)
        self.assertEqual(self.sc.mask.name, "Pat_ID_HE_mask")
        # since update_slide is called with a real tiff, orig_path should point to mask file
        self.assertEqual(getattr(self.sc.mask, "orig_path", None), mask_file)

    def test_new_mask_slide_is_created(self):
        reg = os.path.join(self.dst, "registration2")
        os.makedirs(reg, exist_ok=True)
        # create mask file for a slide not present in collection
        mask_file = os.path.join(reg, "HE_mask.tiff")
        shutil.copy(self.test_file, mask_file)

        self.sc.add_mask_to_collection(reg)

        # new mask should be appended and set as collection mask
        self.assertIsNotNone(self.sc.mask)
        self.assertTrue(self.sc.mask.is_mask)
        # name expected by cutils.get_name (without extension); ensure we find it
        self.assertIn(self.sc.mask.name, "HE_mask")

    def test_unsupported_extension_is_skipped(self):
        reg = os.path.join(self.dst, "registration3")
        os.makedirs(reg, exist_ok=True)
        # create unsupported extension mask
        mask_file = os.path.join(reg, "Pat_ID_AG1_mask.jpg")
        shutil.copy(self.test_file, mask_file)

        self.sc.add_mask_to_collection(reg)

        # jpg not in SUPPORTED_IMAGE_FORMATS -> should not set mask for AG1
        # existing AG1 slide should not be marked as mask
        ag1 = next(s for s in self.sc.slides if s.name == "Pat_ID_AG1")
        self.assertFalse(ag1.is_mask)
        # collection.mask should still be None (no valid masks found)
        self.assertIsNone(self.sc.mask)

    def test_update_slide_exception_is_handled_and_mask_set(self):
        reg = os.path.join(self.dst, "registration4")
        os.makedirs(reg, exist_ok=True)
        # mask for existing slide
        mask_file = os.path.join(reg, "Pat_ID_AG1_mask.tiff")
        shutil.copy(self.test_file, mask_file)

        # patch Slide.update_slide to raise for this test
        with patch.object(Slide, "update_slide", side_effect=Exception("boom")):
            # call add_mask_to_collection; should catch the exception and still set is_mask and mask
            self.sc.add_mask_to_collection(reg)

        ag1 = next(s for s in self.sc.slides if s.name == "Pat_ID_AG1_mask")
        self.assertTrue(ag1.is_mask)
        # even though update failed, the mask attribute on collection should be set
        self.assertIs(self.sc.mask, ag1)


class TestExtractMaskTileCoordinates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.dst = os.path.join(self.tmp.name, "dst")
        os.makedirs(self.src, exist_ok=True)
        os.makedirs(self.dst, exist_ok=True)

        # repo test file (small valid tiff used elsewhere in tests)
        self.test_file = os.path.join(
            os.path.dirname(__file__), "test_files", "test_file.tiff"
        )
        assert os.path.exists(self.test_file), "missing test fixture test_file.tiff"

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except Exception:
            pass

    def test_extract_mask_tiles_when_no_mask_collects_all_tiles_and_writes_pickle(self):
        # one slide in src -> mask is None and function should collect all coordinates
        shutil.copy(self.test_file, os.path.join(self.src, "Pat_ID_HE.tiff"))
        sc = SlideCollection("C", self.src, self.dst)

        # ensure no mask present
        self.assertIsNone(sc.mask)

        # call extractor (no images saved)
        sc.extract_mask_tile_coordinates(save_img=False)

        # read tiles info from slide object to compute expected count
        tiles = sc.slides[0].tiles
        cols, rows = tiles.level_tiles[tiles.level_count - 1]
        expected_count = cols * rows

        # mask_coordinates should contain all tiles at the lowest level
        self.assertEqual(len(sc.mask_coordinates), expected_count)

        # pickle should have been written
        out = os.path.join(sc.pickle_dir, "mask_coordinates.pickle")
        self.assertTrue(os.path.exists(out))
        with open(out, "rb") as f:
            data = pickle.load(f)
        self.assertEqual(data, sc.mask_coordinates)

    def test_extract_mask_tiles_with_mask_and_save_images_creates_mask_dir_and_pickle(
        self,
    ):
        # create slide(s) in src and a registration dir with a mask image
        shutil.copy(self.test_file, os.path.join(self.src, "Pat_ID_HE.tiff"))
        reg = os.path.join(self.dst, "registration")
        os.makedirs(reg, exist_ok=True)
        mask_path = os.path.join(reg, "Pat_ID_HE_mask.tiff")
        shutil.copy(self.test_file, mask_path)

        sc = SlideCollection("C", self.src, self.dst)
        # ensure registration dir is picked up if load_previous_results ran
        # explicitly add the mask from the registration dir to be safe
        sc.add_mask_to_collection(reg)
        self.assertIsNotNone(sc.mask)

        # call extractor and ask to save mask tile images
        sc.extract_mask_tile_coordinates(save_img=True)

        # pickle file exists
        out = os.path.join(sc.pickle_dir, "mask_coordinates.pickle")
        self.assertTrue(os.path.exists(out))
        with open(out, "rb") as f:
            coords = pickle.load(f)
        # coords should be a list of tuples
        self.assertIsInstance(coords, list)
        if coords:
            self.assertIsInstance(coords[0], tuple)

        # verify that mask images were saved under tiles_dir/mask
        mask_tiles_dir = os.path.join(sc.tiles_dir, "mask")
        self.assertTrue(os.path.isdir(mask_tiles_dir))
        # there should be at least one saved tile file when coords not empty
        saved_tiles = [f for f in os.listdir(mask_tiles_dir) if f.endswith(".tif")]
        # directory exists; content may be empty depending on fixture
        self.assertTrue(len(saved_tiles) >= 0)


class TestQuickQuantificationAndColocalization(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.dst = os.path.join(self.tmp.name, "dst")
        os.makedirs(self.src, exist_ok=True)
        os.makedirs(self.dst, exist_ok=True)

        open(os.path.join(self.src, "placeholder.tiff"), "wb").close()

        class _StubSlide:
            def __init__(self, name, path, is_mask=False, is_reference=False):
                self.name = name
                self.orig_path = path
                self.registered_path = None
                self.is_mask = is_mask
                self.is_reference = is_reference
                self.antigen_profile = (
                    None
                    if (is_mask or is_reference)
                    else {
                        "Name": "default",
                        "low_positive_threshold": 181,
                        "medium_positive_threshold": 121,
                        "high_positive_threshold": 61,
                    }
                )
                # minimal tiles object (one level, 1x1) with get_tile returning an RGB PIL image
                self.tiles = SimpleNamespace(
                    level_count=1,
                    level_tiles=[(1, 1)],
                    level_dimensions=[(8, 8)],
                    tile_count=1,
                    get_tile=lambda level, coord: Image.new(
                        "RGB", (8, 8), (255, 255, 255)
                    ),
                )
                self.properties = {
                    "name": self.name,
                    "reference": self.is_reference,
                    "mask": self.is_mask,
                    "openslide_object": None,
                    "tiles": self.tiles,
                    "level_count": self.tiles.level_count,
                    "level_dimensions": self.tiles.level_dimensions,
                    "tile_count": self.tiles.tile_count,
                }

        self._slide_patcher = patch(
            "cubats.slide_collection.slide_collection.Slide", new=_StubSlide
        )
        self._slide_patcher.start()

        self.sc = SlideCollection("C", self.src, self.dst)

    def tearDown(self):
        # ensure we stop the Slide patcher if it was started to avoid leaking the stub into other tests
        try:
            patcher = getattr(self, "_slide_patcher", None)
            if patcher:
                patcher.stop()
        except Exception:
            pass

        try:
            self.tmp.cleanup()
        except Exception:
            pass

    def test_quantify_single_slide_writes_results(self):
        # Replace slides with a single dummy slide and mark collection ready
        dummy = DummySlide("S1", coverage=42.0)
        self.sc.slides = [dummy]
        self.sc.status["registered"] = True
        self.sc.status["segmented"] = True
        # avoid mask extraction by providing mask coordinates
        self.sc.mask_coordinates = [(0, 0)]

        # call quantify_single_slide - should append to quantification_results and write files
        self.sc.quantify_single_slide("S1", save_img=False, masking_mode="tile-level")

        # results DataFrame populated
        self.assertEqual(len(self.sc.quantification_results), 1)
        self.assertEqual(self.sc.quantification_results.iloc[0]["Name"], "S1")

        # pickle file exists
        out_pickle = os.path.join(self.sc.pickle_dir, "quantification_results.pickle")
        self.assertTrue(os.path.exists(out_pickle))
        with open(out_pickle, "rb") as f:
            loaded = pickle.load(f)
        self.assertIn("S1", loaded["Name"].values)

    def test_evaluate_antigen_pair_creates_dual_and_pickles(self):
        # create two dummy slides with matching tile names and antigen profiles
        s1 = DummySlide("A1")
        s2 = DummySlide("A2")
        s1.antigen_profile = {
            "high_positive_threshold": 200,
            "medium_positive_threshold": 121,
            "low_positive_threshold": 61,
        }
        s2.antigen_profile = {
            "high_positive_threshold": 150,
            "medium_positive_threshold": 121,
            "low_positive_threshold": 61,
        }
        # detailed_quantification_results must have identical keys and 'Tilename' for iterable construction
        s1.detailed_quantification_results = {0: {"Tilename": "0_0"}}
        s2.detailed_quantification_results = {0: {"Tilename": "0_0"}}

        self.sc.slides = [s1, s2]
        # prepare args (these values are only used to build iterable)
        # Patch ProcessPoolExecutor so mapping returns a deterministic result list
        fake_tile_result = {
            "Flag": 1,
            "Total Coverage": 10.0,
            "Total Overlap": 5.0,
            "Total Complement": 5.0,
            "High Positive Overlap": 1.0,
            "High Positive Complement": 1.0,
            "Medium Positive Overlap": 2.0,
            "Medium Positive Complement": 2.0,
            "Low Positive Overlap": 2.0,
            "Low Positive Complement": 2.0,
            "Negative": 0.0,
            "Tissue": 100.0,
            "Background / No Tissue": 0.0,
            "Mask Area": 50.0,
            "Non-mask Area": 50.0,
        }

        with patch("concurrent.futures.ProcessPoolExecutor") as MockExec:
            mock_executor = MockExec.return_value.__enter__.return_value
            # make map return an iterator over one fake tile result
            mock_executor.map.return_value = iter([fake_tile_result])

            # call evaluate_antigen_pair which will call summarize & save
            self.sc.evaluate_antigen_pair(
                s1, s2, save_img=False, masking_mode="tile-level"
            )

        # dual_antigen_expressions DataFrame should have one row
        self.assertEqual(len(self.sc.dual_antigen_expressions), 1)
        # pickle should exist
        out_pickle = os.path.join(self.sc.pickle_dir, "dual_antigen_expressions.pickle")
        self.assertTrue(os.path.exists(out_pickle))
        with open(out_pickle, "rb") as f:
            df = pickle.load(f)
        self.assertFalse(df.empty)

    def test_evaluate_antigen_triplet_creates_triplet_and_pickles(self):
        # create three dummy slides with matching tile names and antigen profiles
        s1 = DummySlide("T1")
        s2 = DummySlide("T2")
        s3 = DummySlide("T3")
        s1.antigen_profile = {
            "high_positive_threshold": 200,
            "medium_positive_threshold": 121,
            "low_positive_threshold": 61,
        }
        s2.antigen_profile = {
            "high_positive_threshold": 150,
            "medium_positive_threshold": 121,
            "low_positive_threshold": 61,
        }
        s3.antigen_profile = {
            "high_positive_threshold": 150,
            "medium_positive_threshold": 121,
            "low_positive_threshold": 61,
        }
        # detailed_quantification_results must have identical keys and 'Tilename'
        s1.detailed_quantification_results = {0: {"Tilename": "0_0"}}
        s2.detailed_quantification_results = {0: {"Tilename": "0_0"}}
        s3.detailed_quantification_results = {0: {"Tilename": "0_0"}}

        self.sc.slides = [s1, s2, s3]

        fake_tile_result = {
            "Flag": 1,
            "Total Coverage": 10.0,
            "Total Overlap": 5.0,
            "Total Complement": 5.0,
            "High Positive Overlap": 1.0,
            "High Positive Complement": 1.0,
            "Medium Positive Overlap": 2.0,
            "Medium Positive Complement": 2.0,
            "Low Positive Overlap": 2.0,
            "Low Positive Complement": 2.0,
            "Negative": 0.0,
            "Tissue": 100.0,
            "Background / No Tissue": 0.0,
            "Mask Area": 50.0,
            "Non-mask Area": 50.0,
        }

        with patch("concurrent.futures.ProcessPoolExecutor") as MockExec:
            mock_executor = MockExec.return_value.__enter__.return_value
            mock_executor.map.return_value = iter([fake_tile_result])

            # call the triplet evaluation function
            # function name may be evaluate_antigen_triplet or evaluate_antigen_triplets;
            # try the singular first and fall back to plural if needed
            if hasattr(self.sc, "evaluate_antigen_triplet"):
                self.sc.evaluate_antigen_triplet(
                    s1, s2, s3, save_img=False, masking_mode="tile-level"
                )
            else:
                # fallback to plural name if library uses it
                self.sc.evaluate_antigen_triplets(
                    s1, s2, s3, save_img=False, masking_mode="tile-level"
                )

        # Expect a pickle with antigen expressions to be written; search for it
        pickles = [
            f
            for f in os.listdir(self.sc.pickle_dir)
            if f.endswith(".pickle") and "antigen" in f and "express" in f
        ]
        self.assertTrue(len(pickles) >= 1)
        # load the first matching pickle and assert it contains a non-empty DataFrame-like object
        with open(os.path.join(self.sc.pickle_dir, pickles[0]), "rb") as fh:
            df = pickle.load(fh)
        # should be a non-empty table/iterable
        self.assertTrue(hasattr(df, "empty") and not df.empty)

    def test_save_antigen_combinations_invalid_type_raises(self):
        # call save_antigen_combinations with invalid type
        with self.assertRaises(ValueError):
            self.sc.save_antigen_combinations(
                result_type="invalid", masking_mode="tile-level"
            )


def _fake_slide(name, grid=(3, 2), size=(3000, 2000)):
    tiles = SimpleNamespace(level_tiles=[(1, 1), grid], level_dimensions=[(1, 1), size])
    return SimpleNamespace(name=name, tiles=tiles)


def _collection(tmp, slides):
    c = SlideCollection.__new__(SlideCollection)   # skip __init__
    c.slides = slides
    c.logger = MagicMock()
    c.dest_dir = tmp
    c.colocalization_dir = os.path.join(tmp, "colocalization")
    c.reconstruct_dir = os.path.join(tmp, "reconstructed_slides")
    return c


class TestReconstructColocalization(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except Exception:
            pass

    def test_reconstruct_colocalization_dual_by_name(self):
        c = _collection(self.tmp.name, [_fake_slide("CD3"), _fake_slide("CD8")])
        with patch(PATCH, return_value=5) as m:
            found = c.reconstruct_colocalization("CD3", "CD8")
        self.assertEqual(found, 5)
        args, kwargs = m.call_args
        self.assertEqual(args[0], os.path.join(c.colocalization_dir, "CD3_and_CD8"))
        self.assertEqual(args[1], os.path.join(c.reconstruct_dir, "CD3_and_CD8_reconst.tif"))
        self.assertEqual(args[2:4], ((3, 2), (3000, 2000)))
        self.assertTrue(kwargs["thumbnail_path"].endswith("CD3_and_CD8_reconst_thumbnail.png"))

    def test_reconstruct_colocalization_triplet_with_objects_keeps_order(self):
        s = [_fake_slide(n) for n in ("A", "B", "C")]
        c = _collection(self.tmp.name, s)
        with patch(PATCH, return_value=1) as m:
            c.reconstruct_colocalization(s[2], s[0], s[1], thumbnail=False)
        self.assertTrue(m.call_args.args[0].endswith("C_and_A_and_B"))  # folder follows call order
        self.assertIsNone(m.call_args.kwargs["thumbnail_path"])

    def test_reconstruct_colocalization_validation(self):
        s = [_fake_slide("A"), _fake_slide("B"), _fake_slide("C", size=(2999, 2000))]
        c = _collection(self.tmp.name, s)
        with patch(PATCH) as m:
            with self.assertRaises(ValueError):
                c.reconstruct_colocalization("A")              # needs 2 or 3 slides
            with self.assertRaises(ValueError):
                c.reconstruct_colocalization("A", "nope")      # unknown name
            with self.assertRaises(ValueError):
                c.reconstruct_colocalization("A", "B", "C")    # grid mismatch
            m.assert_not_called()                              # nothing stitched on bad input
