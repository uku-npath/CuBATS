# Standard Library
import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Third Party
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# CuBATS
from cubats import logging_config
from cubats.slide_collection import segmentation as seg
from cubats.slide_collection.segmentation import (_save_segmented_wsi,
                                                  _save_thumbnail,
                                                  _segment_file,
                                                  run_tumor_segmentation)


class BaseSegmentationTest(unittest.TestCase):
    """Shared setup/teardown and helpers for segmentation tests."""

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        # initialize logging once per test
        logging.config.dictConfig(logging_config.LOGGING)
        self.default_logger = logging.getLogger(
            "cubats.slide_collection.segmentation")

    def tearDown(self):
        try:
            self.test_dir.cleanup()
        except Exception:
            pass

    # helpers
    def make_image(self, rel_path, size=(512, 512), mode="RGB"):
        p = os.path.join(self.test_dir.name, rel_path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        Image.new(mode, size).save(p)
        return p

    def make_file(self, rel_path, content="data"):
        p = os.path.join(self.test_dir.name, rel_path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(content)
        return p


class TestRunTumorSegmentation(BaseSegmentationTest):
    def setUp(self):
        super().setUp()
        # Path to a valid image
        self.valid_image_path = self.make_image("ref_norm.png", (512, 512))
        # Mock valid model path
        self.valid_model_path = self.make_file(
            "model.onnx", "mock model content")
        # invalid inputs
        self.invalid_input_path = os.path.join(
            self.test_dir.name, "nonexistent.png")
        self.invalid_model_path = self.make_file(
            "invalid_model.onnx", "invalid model content")
        self.invalid_output_path = os.path.join(
            self.test_dir.name, "invalid_output")
        # ensure logs dir exists
        os.makedirs(os.path.join(self.test_dir.name, "logs"), exist_ok=True)
        self.logger = logging.getLogger(__name__)

    @patch("onnx.checker.check_model")
    @patch("onnx.load", side_effect=FileNotFoundError("Model file not found"))
    def test_invalid_model_path(self, mock_load, mock_check_model):
        with self.assertRaises(ValueError) as context:
            run_tumor_segmentation(
                input_path=self.valid_image_path,
                model_path=self.invalid_model_path,
                tile_size=(256, 256),
                output_path=None,
                normalization=False,
                inversion=False,
                plot_results=False,
            )
        self.assertIn("is not a valid path", str(context.exception))

    @patch("onnx.checker.check_model")
    @patch("onnx.load")
    @patch("onnx2torch.convert")
    def test_invalid_input_path(self, mock_convert, mock_load, mock_check_model):
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_convert.return_value = mock_model
        mock_check_model.return_value = None
        with self.assertRaises(FileNotFoundError) as context:
            run_tumor_segmentation(
                input_path=self.invalid_input_path,
                model_path=self.valid_model_path,
                tile_size=(256, 256),
                output_path=None,
                normalization=False,
                inversion=False,
                plot_results=False,
            )
        self.assertIn("Input path", str(context.exception))

    @patch("onnx.checker.check_model")
    @patch("onnx.load")
    @patch("onnx2torch.convert")
    def test_invalid_output_path(self, mock_convert, mock_load, mock_check_model):
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_convert.return_value = mock_model
        mock_check_model.return_value = None
        with self.assertRaises(FileNotFoundError) as context:
            run_tumor_segmentation(
                input_path=self.valid_image_path,
                model_path=self.valid_model_path,
                tile_size=(256, 256),
                output_path=self.invalid_output_path,
                normalization=False,
                inversion=False,
                plot_results=False,
            )
        self.assertIn("Output path", str(context.exception))

    @patch("onnx.checker.check_model")
    @patch("onnx.load")
    @patch("onnx2torch.convert")
    def test_invalid_output_path_not_directory(self, mock_convert, mock_load, mock_check_model):
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_convert.return_value = mock_model
        mock_check_model.return_value = None

        with open(self.invalid_output_path, "w") as f:
            f.write("This is a file, not a directory.")
        with self.assertRaises(ValueError) as context:
            run_tumor_segmentation(
                input_path=self.valid_image_path,
                model_path=self.valid_model_path,
                tile_size=(256, 256),
                output_path=self.invalid_output_path,
                normalization=False,
                inversion=False,
                plot_results=False,
            )
        self.assertIn("is not a directory", str(context.exception))

    @patch("onnx.checker.check_model")
    @patch("onnx.load")
    @patch("cubats.slide_collection.segmentation.convert", side_effect=RuntimeError("Error converting model"))
    def test_model_conversion_error(self, mock_convert, mock_load, mock_check_model):
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_model.graph.input[0].type.tensor_type.shape.dim = [
            MagicMock(dim_value=1),
            MagicMock(dim_value=3),
            MagicMock(dim_value=224),
            MagicMock(dim_value=224),
        ]
        mock_check_model.return_value = None
        with self.assertRaises(RuntimeError) as context:
            run_tumor_segmentation(
                input_path=self.valid_image_path,
                model_path=self.valid_model_path,
                tile_size=(256, 256),
                output_path=None,
                normalization=False,
                inversion=False,
                plot_results=False,
            )
        self.assertIn("Error converting model", str(context.exception))


class TestSegmentFile(BaseSegmentationTest):
    def setUp(self):
        super().setUp()
        self.valid_wsi_path = self.make_image("valid_wsi.tif", (512, 512))
        self.invalid_wsi_path = os.path.join(
            self.test_dir.name, "invalid_wsi.tif")
        self.output_path = os.path.join(self.test_dir.name, "output")
        os.makedirs(self.output_path, exist_ok=True)
        self.logger = logging.getLogger("cubats.slide_collection.segmentation")

    @patch("openslide.OpenSlide")
    def test_open_slide_error(self, mock_open_slide):
        mock_open_slide.side_effect = Exception("Error opening slide file")
        with self.assertLogs(self.logger, level="ERROR") as log:
            _segment_file(
                self.invalid_wsi_path,
                MagicMock(),
                (256, 256),
                (1, 3, 224, 224),
                self.output_path,
                normalization=False,
                inversion=False,
                plot_results=False,
            )
            self.assertTrue(len(log.output) > 0)
            self.assertIn("Error opening slide file", log.output[0])


class TestSegmentFileProcessing(BaseSegmentationTest):
    def setUp(self):
        super().setUp()
        self.wsi_path = self.make_image("valid_wsi_small.tif", (32, 16))
        self.output_path = os.path.join(self.test_dir.name, "out")
        os.makedirs(self.output_path, exist_ok=True)

    def test_segment_file_calls_save_functions(self):
        tile_size = (16, 16)

        class DummyDZG:
            def __init__(self, slide, tile_size, overlap=0, limit_bounds=False):
                self.level_tiles = [(1, 1), (2, 1)]
                self.level_count = 2
                self._tile_size = tile_size

            def get_tile(self, lvl, coord):
                return Image.new("RGB", (tile_size[0], tile_size[1]), (128, 128, 128))

        class DummyModel(nn.Module):
            def forward(self, x):
                n, _, h, w = x.shape
                return torch.full((n, 1, h, w), 10.0)

        model = DummyModel()

        with patch.object(seg, "OpenSlide", return_value=MagicMock(dimensions=(32, 16))), patch.object(
            seg,
            "DeepZoomGenerator",
            side_effect=lambda slide, tile_size, overlap=0, limit_bounds=False: DummyDZG(
                slide, tile_size
            ),
        ), patch.object(seg, "_init_normalizer", return_value=MagicMock()), patch.object(
            seg, "_save_segmented_wsi", return_value=None
        ) as mock_save_wsi, patch.object(seg, "_save_thumbnail", return_value=None) as mock_save_thumb:
            seg._segment_file(
                self.wsi_path,
                model,
                tile_size,
                (1, 3, tile_size[0], tile_size[1]),
                self.output_path,
                normalization=False,
                inversion=False,
                plot_results=False,
            )

            self.assertTrue(mock_save_wsi.called,
                            "_save_segmented_wsi was not called")
            self.assertTrue(mock_save_thumb.called,
                            "_save_thumbnail was not called")
            called_args = mock_save_wsi.call_args[0]
            if len(called_args) >= 3:
                mask_out = called_args[2]
                self.assertIn("_mask", os.path.basename(mask_out))
                self.assertTrue(mask_out.endswith(".tif"))


class TestRunBatch(BaseSegmentationTest):
    def _run(self, model, needs_resize, inversion, tile_size=(8, 8)):
        output_tiles = [None]
        pbar = MagicMock()
        seg._run_batch(
            batch_coords=[(0, 0)],
            batch_tensors=[torch.zeros((3, 8, 8), dtype=torch.float32)],
            model=model,
            device=torch.device("cpu"),
            use_amp=False,
            needs_resize=needs_resize,
            tile_size=tile_size,
            inversion=inversion,
            output_tiles=output_tiles,
            n_cols=1,
            pbar=pbar,
        )
        return output_tiles, pbar

    def test_run_batch_inversion(self):
        class NegModel(nn.Module):
            def forward(self, x):
                n, _, h, w = x.shape
                return torch.full((n, 1, h, w), -10.0)

        tiles, pbar = self._run(NegModel(), needs_resize=False, inversion=True)
        self.assertEqual(tiles[0].shape, (8, 8))
        self.assertEqual(tiles[0].dtype, np.uint8)
        self.assertTrue((tiles[0] == 255).all())
        pbar.update.assert_called_once_with(1)

    def test_run_batch_resizing(self):
        class SmallModel(nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 1, 4, 4), 10.0)

        tiles, _ = self._run(SmallModel(), needs_resize=True, inversion=False)
        self.assertEqual(tiles[0].shape, (8, 8))
        self.assertTrue((tiles[0] == 255).all())

    def test_run_batch_empty_is_noop(self):
        output_tiles, pbar = [None], MagicMock()
        seg._run_batch([], [], MagicMock(), torch.device("cpu"), False,
                       False, (8, 8), False, output_tiles, 1, pbar)
        self.assertIsNone(output_tiles[0])
        pbar.update.assert_not_called()


class TestSaveSegmentedWSI(BaseSegmentationTest):
    def setUp(self):
        super().setUp()
        self.output_path = os.path.join(
            self.test_dir.name, "segmented_wsi.tif")
        self.segmented_wsi = MagicMock()
        self.segmented_wsi.width = 512
        self.segmented_wsi.height = 512

    @patch("pyvips.Image.crop", return_value=MagicMock())
    def test_save_segmented_wsi_success(self, mock_crop):
        try:
            _save_segmented_wsi(self.segmented_wsi,
                                (256, 256), self.output_path)
        except Exception as e:
            self.fail(f"_save_segmented_wsi raised an exception: {e}")


class TestSaveThumbnail(BaseSegmentationTest):
    def setUp(self):
        super().setUp()
        self.wsi_path = self.make_image("wsi.tif", (512, 512))
        self.output_path = os.path.join(self.test_dir.name, "thumbnail.png")

    @patch("openslide.OpenSlide.get_thumbnail")
    @patch("openslide.OpenSlide")
    def test_save_thumbnail_success(self, mock_openslide, mock_get_thumbnail):
        mock_slide = MagicMock()
        mock_openslide.return_value = mock_slide
        mock_thumbnail = Image.new("RGB", (512, 512))
        mock_get_thumbnail.return_value = mock_thumbnail
        try:
            _save_thumbnail(self.wsi_path, self.output_path)
        except Exception as e:
            self.fail(f"_save_thumbnail raised an exception: {e}")

    @patch("openslide.OpenSlide", side_effect=Exception("Error opening WSI"))
    def test_save_thumbnail_error(self, mock_openslide):
        with self.assertLogs("cubats.slide_collection.segmentation", level="ERROR") as log:
            _save_thumbnail(self.wsi_path, self.output_path)
            self.assertIn("Error saving PNG thumbnail", log.output[0])


if __name__ == '__main__':
    unittest.main()
