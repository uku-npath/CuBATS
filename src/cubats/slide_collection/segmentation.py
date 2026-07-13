# Standard Library
import logging.config
import os
from os import listdir, path
from time import time
from typing import Tuple, Union

# Third Party
import numpy as np
import onnx
import torch
import torch.nn.functional as F
import torchstain
import torchvision
from numpy import concatenate
from onnx2torch import convert
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from pyvips import BandFormat
from pyvips import Image as VipsImage
from skimage.transform import resize
from tqdm import tqdm

# CuBATS
import cubats.logging_config as log_config
from cubats import cutils as cutils

# Initialize logging
logging.config.dictConfig(log_config.LOGGING)
logger = logging.getLogger(__name__)
# Suppress pyvips logs
logging.getLogger("pyvips").setLevel(logging.ERROR)

# Currently only works for pytorch input order, as some steps are hardcoded and onnx2torch is used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())
# logger.info(f"Using device: {device}")


def run_tumor_segmentation(
    input_path: str,
    model_path: str,
    tile_size: Tuple[int, int],
    output_path: Union[str, None] = None,
    normalization: bool = False,
    inversion: bool = False,
    plot_results: bool = False,
):
    """Run the segmentation pipeline on the given input path using the specified model.

    Performs segmentation on a single HE stained WSI or all HE stained WSIs in a directory using the specified model.
    The segmentation results are saved as .TIFF in the output directory, as well as a . PNG thumbnail. If no output
    directory is provided, the results are saved in the same directory as the input. Optionally, a thumbnail of the
    segmentation results can be plotted on the original image and saved.

    Args:
        input_path (str): The path to the input file or directory.
        model_path (str): The path to the ONNX model file.
        tile_size (Tuple[int, int]): The size of each tile for segmentation.
        output_path (Union[str, None], optional): The path to the output directory. If not provided, the output will be
            saved in the same directory as the input. Defaults to None.
        normalization (bool, optional): Whether to normalize the input tiles. Depends on the model provided. Defaults
            to False.
        inversion (bool, optional): Whether to invert the segmentation output. Depends on the model provided. Defaults
            to False.
        plot_results (bool, optional): Whether to plot the segmentation results. Defaults to False.

    Raises:
        FileNotFoundError: If the input path or output path does not exist.
        ValueError: If the output path is not a directory or the model path is invalid.

    Returns:
        None
    """
    logger.info(
        f"Starting segmentation of: {path.splitext(path.basename(input_path))[0]}; "
        f"using model: {path.splitext(path.basename(model_path))[0]}; "
        f"Parameters: tile_size: {tile_size}, normalization: {normalization}, "
        f"inversion: {inversion}, plot_results: {plot_results}"
    )

    start_time_segmentation = time()
    # Check if the input path is valid and if it is a file or a directory
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path {input_path} does not exist.")
        if os.path.isfile(input_path):
            segment_single_file = True
            input_folder = os.path.dirname(input_path)
        else:
            segment_single_file = False
            input_folder = input_path
    except FileNotFoundError as e:
        logger.error(e)
        raise
    except Exception as e:
        logger.error(f"Unexpected error checking input path: {e}")
        raise

    if path.isfile(input_path):
        segment_single_file = True
        input_folder = path.dirname(input_path)
    else:
        segment_single_file = False
        input_folder = input_path

    # Check the output path and raise an error if it is not a path or a directory
    if output_path is None:
        output_path = input_folder
    else:
        if not path.exists(output_path):
            logger.error(f"Output path {output_path} does not exist.")
            raise FileNotFoundError(f"Output path {output_path} does not exist.")
        if not path.isdir(output_path):
            logger.error(f"Output path {output_path} is not a directory.")
            raise ValueError(f"Output path {output_path} is not a directory.")

    # Check if the model is an valid onnx file
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
    except ValueError as e:
        logger.error(f"Model {model_path} is not a valid onnx file.")
        raise e
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        raise ValueError(f"Model {model_path} is not a valid path.")

    try:
        model_input_size = _get_model_input_size(model)
    except Exception as e:
        logger.error(f"Error getting model input size: {e}")
        raise RuntimeError(f"Error getting model input size: {e}")

    try:
        model = convert(model)
        # model.to(device)
    except Exception as e:
        logger.error(f"Error converting model {model_path}: {e}")
        raise RuntimeError(f"Error converting model {model_path}: {e}")
    # try:
    #     session_options = ort.SessionOptions()
    #     session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    #     session_options.graph_optimization_level = (
    #         ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    #     )
    #     providers = (
    #         ["CUDAExecutionProvider"]
    #         if ort.get_device() == "GPU"
    #         else ["CPUExecutionProvider"]
    #     )
    #     ort_session = ort.InferenceSession(
    #         model_path, sess_options=session_options, providers=providers
    #     )
    # except Exception as e:
    #     logger.error(f"Error creating ONNX Runtime session: {e}")
    #     raise RuntimeError(f"Error creating ONNX Runtime session: {e}")

    # Suppress Pillow warning because of the large WSI
    Image.MAX_IMAGE_PIXELS = None

    # Run tumor segmentation
    if segment_single_file:
        logger.info("Segementation-mode: single file")
        _segment_file(
            input_path,
            model,
            tile_size,
            model_input_size,
            output_path,
            normalization,
            inversion,
            plot_results,
        )
    else:
        logger.info("Segementation-mode: multiple files")
        for file in listdir(input_folder):
            if file.endswith(".tif"):
                _segment_file(
                    path.join(input_folder, file),
                    model,
                    tile_size,
                    model_input_size,
                    output_path,
                    normalization,
                    inversion,
                    plot_results,
                )
        logger.info(f"Segmentation of all files in {input_folder} completed.")

    end_time_segmentation = time()
    logger.info(
        f"Segmentation of {input_path} completed in"
        f"{(end_time_segmentation - start_time_segmentation) / 60: .2f} Minutes."
    )


def _segment_file(
    file_path,
    model,
    tile_size,
    model_input_size,
    output_path,
    normalization,
    inversion,
    plot_results,
):
    """Performs tumor detection and segmentation on a single WSI file.

    Segments a (WSI) file into tiles, applies a tumor segmentation model onto each tile. The resulting tiles are
    reconstructed into a single image and saved as a .TIFF file. A thumbnail of the segmentation results is saved as a
    .PNG file. Optionally, the segmentation results can be plotted onto the original image and saved.

    Args:
        file_path (str): The path to the WSI file.
        model: The segmentation model.
        tile_size (tuple): The size of each tile in pixels.
        model_input_size (tuple): The input size of the model.
        output_path (str): The path to save the segmented image.
        normalization (bool): Whether to normalize the input tiles.
        inversion (bool): Whether to invert the segmentation output.
        plot_results (bool): Whether to plot the segmentation results on the original image.

    Returns:
        None
    """
    logger.info(f"Starting segmentation for file: {file_path}")
    try:
        slide = OpenSlide(file_path)
        logger.debug(f"Opened slide file: {file_path}")
    except Exception as e:
        logger.error(f"Error opening slide file {file_path}: {e}")
        return

    try:
        slide_generator = DeepZoomGenerator(
            slide, tile_size=tile_size[0], overlap=0, limit_bounds=False
        )
        logger.debug("Created DeepZoomGenerator")
    except Exception as e:
        logger.error(f"Error creating DeepZoomGenerator: {e}")
        return

    # Initialize the normalizer. Reference image is provided in assets folder
    if normalization:
        ref_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "assets", "ref_norm.png")
        )
        normalizer = _init_normalizer(ref_path)

    # Calculate if there is an overhang for the final tile at the right and bottom edge
    has_right_overhang = (
        slide_generator.level_tiles[-1][0] * tile_size[0] - slide.dimensions[0]
    ) > 0
    has_bottom_overhang = (
        slide_generator.level_tiles[-1][1] * tile_size[1] - slide.dimensions[1]
    ) > 0

    # Array to save the segmented WSI rows as single numpy arrays
    row_array = []

    # Segment each tile
    for row in tqdm(
        range(0, slide_generator.level_tiles[-1][1]),
        desc=f"Segmenting Rows for {path.splitext(path.basename(file_path))[0]}",
    ):
        logger.debug(f"Processing row {row}")
        # Per default no padding is required
        is_rightmost_tile = False

        # Array to save the segmented tiles of all columns of a row
        column_array = []

        # Check if the tile is the rightmost tile
        if row == slide_generator.level_tiles[-1][1] - 1 and has_right_overhang:
            is_rightmost_tile = True

        for column in range(0, slide_generator.level_tiles[-1][0]):
            is_bottommost_tile = False

            # Check if the tile is the bottommost tile
            if column == slide_generator.level_tiles[-1][0] - 1 and has_bottom_overhang:
                is_bottommost_tile = True

            # If the tile is the rightmost or bottommost tile, padding is required
            if is_rightmost_tile or is_bottommost_tile:
                raw_tile = slide_generator.get_tile(
                    slide_generator.level_count - 1, (column, row)
                )
                tile = Image.new("RGB", tile_size, (255, 255, 255))
                tile.paste(raw_tile, (0, 0))
                logger.debug(f"Tile at row {row}, column {column} required padding")
            else:
                tile = slide_generator.get_tile(
                    slide_generator.level_count - 1, (column, row)
                )

            # Normalize the tensor if needed
            if normalization:
                # Convert the tile to a tensor
                transform = torchvision.transforms.ToTensor()
                tile_tensor = transform(tile).float() * 255

                normalized_tile = normalizer.normalize(tile_tensor)
                # Recreate tensor with correct shape, including batch dimension
                tile_tensor = (
                    normalized_tile.cpu().permute(2, 0, 1).unsqueeze(0).float()
                )

                if tile_size != model_input_size[2:]:
                    resizing = True
                    # Resize the tile to the model input size if needed using torch
                    tile_tensor = F.interpolate(
                        tile_tensor,
                        size=model_input_size[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    resizing = False
            else:
                # Resize the tile to the model input size if needed using numpy.
                # Note: Model used during developement needed numpy conversion as torch resulted in false results.
                if tile_size != model_input_size[2:]:
                    resizing = True
                    rescaled_tile = resize(
                        np.asarray(tile),
                        (model_input_size[2], model_input_size[3]),
                        anti_aliasing=True,
                    )
                    # Convert the tile to a tensor
                    tile_tensor = (
                        torch.from_numpy(rescaled_tile)
                        .type(torch.float32)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
                else:
                    resizing = False
                    # Convert the tile to a tensor and add batch dimension
                    transform = torchvision.transforms.ToTensor()
                    tile_tensor = transform(tile).float() * 255
                    tile_tensor.unsqueeze_(0)

            # tile_tensor = tile_tensor.to(device)
            # Segment the tile
            segmented_tile = _segment_tile(
                tile_tensor, model, resizing, inversion, tile_size
            )
            column_array.append(segmented_tile)

        segmented_row = concatenate(column_array, axis=1)
        row_array.append(segmented_row)

    logger.debug("Constructing segmented WSI")
    segmented_wsi = concatenate(row_array, axis=0)
    segmented_wsi = VipsImage.new_from_array(segmented_wsi).cast(BandFormat.INT)

    # Extract the base name and file type
    base_name, file_type = path.splitext(file_path)
    wsi_name = cutils.get_name(base_name)

    # Construct output paths for the mask and thumbnail
    mask_out = path.join(output_path, f"{wsi_name}_mask{file_type}")
    thumb_out = path.join(output_path, f"{wsi_name}_mask_thumbnail.png")

    # Save the segmented WSI using pyvips
    _save_segmented_wsi(segmented_wsi, tile_size, mask_out)

    # Create and save PNG thumbnail
    _save_thumbnail(mask_out, thumb_out)

    # if plot_results: save cropped segmentation result on top of original image
    if plot_results:
        try:
            _plot_segmentation_on_tissue(file_path, output_path)
            logger.debug("Plotted segmentation on tissue")
        except Exception as e:
            logger.error(f"Error plotting segmentation on tissue: {e}")

    logger.info(f"Finished segmentation for file: {file_path}")


def _segment_tile(
    tile: torch.Tensor, model, resizing: bool, inversion: bool, original_size
) -> Image:
    """
    Segments a given tile using a pre-trained model and returns the segmented tile as a PIL Image.

    Args:
        tile (torch.Tensor): The input tile to be segmented. Expected shape is (1, C, H, W).
        model: The pre-trained segmentation model.
        resizing (bool): Whether input has been resized to the model's input size. Consequentially resizing the output
        is necessary.
        inversion (bool): Whether to invert the segmentation output, depending on model type.
        original_size (tuple): The original size of the tile before resizing.

    Returns:
        Image: The segmented tile as a binary mask in PIL Image format.
    """
    # Start segmentation
    with torch.no_grad():
        segmentation = model(tile)

    segmentation = segmentation.sigmoid()

    # Check the shape and apply squeeze as needed
    if segmentation.dim() == 4 and segmentation.shape[1] == 1:
        # Remove single color channel dimension
        segmentation = segmentation.squeeze(1)

    segmentation = segmentation.squeeze(0).cpu().numpy()  # Remove batch dimension

    # Rescale to original size if necessary
    if resizing:
        segmented_tile = resize(segmentation, original_size, anti_aliasing=True)
    else:
        segmented_tile = segmentation

    # Some models require inversion
    if inversion:
        segmented_tile = 1 - segmented_tile

    # Threshold to binary mask
    segmented_tile = (segmented_tile > 0.5).astype(np.uint8)

    # Convert to PIL image
    segmented_tile_pil = Image.fromarray(segmented_tile * 255, mode="L")

    return segmented_tile_pil


def _init_normalizer(path_to_src_img):
    """Initializes a Reinhard normalizer for image normalization.

    Args:
         path_to_src_img (str): Path to the source image file.

    Returns:
        torchstain.normalizers.ReinhardNormalizer: An instance of the Reinhard normalizer fitted to the source image.
    """
    normalizer = torchstain.normalizers.ReinhardNormalizer(
        method="modified", backend="torch"
    )
    src_img = torchvision.io.read_image(path_to_src_img)
    normalizer.fit(src_img)
    return normalizer


def _get_model_input_size(model):
    """
    Get the expected input size for the model.

    Args:
        model: The pre-trained segmentation model.

    Returns:
        tuple: The expected input size for the model in the format (1, C, H, W).
    """
    input_tensor = model.graph.input[0]
    input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    # Ensure the batch size is set to 1
    input_shape[0] = 1
    logger.debug(f"Model input size: {input_shape}")
    return tuple(input_shape)


def _save_segmented_wsi(segmented_wsi, tile_size, output_path):
    """Save the segmented WSI to file.
    Args:
        segmented_wsi (pyvips.Image): The segmented WSI image.
        output_path (str): The path to save the segmented WSI.
    """
    logger.info(f"Saving segmented WSI to {output_path}")
    try:
        segmented_wsi.crop(0, 0, segmented_wsi.width, segmented_wsi.height).tiffsave(
            output_path,
            tile=True,
            compression="jpeg",
            bigtiff=True,
            pyramid=True,
            tile_width=tile_size[0],
            tile_height=tile_size[1],
            Q=100,  # Set JPEG quality to 100%
            predictor="horizontal",  # Use horizontal predictor for better compression
            strip=True,  # Strip metadata to reduce file size
        )
        logger.debug(f"Segmented WSI saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving segmented WSI: {e}")


def _save_thumbnail(wsi_path, output_path):
    """Create and save PNG thumbnail.

    Args:
        wsi_path (str): The path to the WSI file, based on which the thumbnail shall be created.
        output_path (str): The path to save the PNG thumbnail.
    """
    logger.info("Creating PNG thumbnail for segmented WSI")
    try:
        wsi = OpenSlide(wsi_path)
        thumbnail = wsi.get_thumbnail((512, 512))
        thumbnail.save(output_path, "PNG")
        logger.debug(f"PNG thumbnail saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving PNG thumbnail: {e}")


def _plot_segmentation_on_tissue(file_path, output_path):
    """
    Plots the segmentation results on the original image and saves the result as a thumbnail.

    Args:
        file_path (str): The path to the original WSI file.
        output_path (str): The path to save the thumbnail.

    Returns:
        None
    """
    start_time = time()
    logger.info("Plotting thumbnail for segmented WSI")

    slide = OpenSlide(file_path)

    wsi_name, file_type = path.splitext(file_path)
    wsi_name = path.splitext(wsi_name)[0] + "_mask" + file_type
    mask_path = path.join(output_path, path.basename(wsi_name))
    mask = OpenSlide(mask_path)

    logger.info("Retrieving thumbnail for slide")
    slide_thumbnail = slide.get_thumbnail(
        (slide.dimensions[0] / 256, slide.dimensions[1] / 256)
    )
    logger.info("Retrieving thumbnail for mask")
    mask_thumbnail = mask.get_thumbnail(
        (mask.dimensions[0] / 256, mask.dimensions[1] / 256)
    )

    # Convert mask to RGBA
    logger.info("Converting mask to RGBA")
    mask_thumbnail = mask_thumbnail.convert("RGBA")

    # Split the mask into its components
    logger.info("Splitting mask into components")
    r, g, b, a = mask_thumbnail.split()

    # Create a new alpha channel where white areas are fully transparent
    logger.info("Creating new alpha channel")
    alpha = Image.eval(a, lambda px: 0 if px == 255 else 255)

    # Combine the mask with the new alpha channel
    logger.info("Combining mask with new alpha channel")
    mask_thumbnail = Image.merge("RGBA", (r, g, b, alpha))

    # Ensure the slide is in RGB mode (no alpha)
    logger.info("Converting slide to RGB")
    slide_thumbnail = slide_thumbnail.convert("RGB")

    # Composite the slide and mask
    logger.info("Compositing slide and mask")
    combined = Image.alpha_composite(slide_thumbnail.convert("RGBA"), mask_thumbnail)

    png_name, _ = path.splitext(file_path)
    png_name = path.splitext(png_name)[0] + "_mask" + ".png"
    logger.info(f"Saving thumbnail to {path.join(output_path, png_name)}")
    combined.save(path.join(output_path, png_name))

    end_time = time()
    logger.debug(f"Thumbnail creation took {end_time - start_time: .2f} seconds.")
