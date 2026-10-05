"""
This module performs tile-level quantification of IHC tiles, including color deconvolution, quantification of staining
intensities, H-score and the IHC-Profiler score calculation, all based on antigen-specific thresholds, and masking of
tumor-tissue areas.
"""

# Standard Library
import time

# Third Party
import cv2
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from skimage import img_as_ubyte
from skimage.color import hed2rgb, rgb2gray, rgb2hed
from skimage.exposure import histogram

# CuBATS
from cubats.config import xp
from cubats.cutils import to_numpy

_state = {}


def quantify_tile(iterable):
    """This function quantifies a single input tile and returns a dictionary.

    The function offers two masking modes: tile-level masking and pixel-level masking. In tile-level masking tiles that
    contain sufficient tissue (mean pixel value < 230 and standard deviation > 15), will undergo stain separation and
    quantification. The results will be returned in the dictionary. The 'flag' will be set to 1. If 'save_img' is True
    the DAB image will additionally be saved in the specified directory. If the tile does not contain sufficient tissue
    it will not be processed and the returned dictionary will only contain the tile name and a 'flag' set to 0.
    In pixel-level masking the mask will be applied to the tile prior to quantification to receive a merged tile which
    will then be quantified. Non-mask pixels will be set to 255.


    Args:
        iterable (iterable): Iterable containing the following Information on passed tile:

            - index 0: Column, necessary for naming.
            - index 1: Row, necessary for naming.
            - index 2: Tile itself, necessary since processes cannot access shared memory.
            - DAB_TILE_DIR: Directory, for saving Image, since single processes cannot access shared memory.
            - save_img: Boolean, if True, DAB image will be saved in specified directory.
            - antigen_profile: Antigen-specific thresholds used during quantification.
            - masking_mode: Masking mode for quantification.

    Returns:
        dict: Dictionary containing tile results:

            - Tilename (str): Name of the tile.
            - Histogram (ndarray): Histogram of the tile.
            - Hist_centers (ndarray): Centers of histogram bins.
            - Zones (ndarray): Number of pixels in each intensity zone.
            - Percentage (ndarray): Percentage of pixels in each zone.
            - Score (ndarray): Score for the tile.
            - Tissue Count (int): Total number of tissue pixels in the tile.
            - Flag (int): Processing flag (1 if processed, 0 if not).
            - Image Array (ndarray): Array of pixel values for positive pixels.
    """
    # Assign local variables for better readability
    col = iterable[0]
    row = iterable[1]
    tile = iterable[2][0]
    mask = iterable[2][1]
    DAB_TILE_DIR = iterable[3]
    save_img = iterable[4]
    antigen_profile = iterable[5]
    # masking_mode = iterable[6]

    tile_name = str(col) + "_" + str(row)

    # Initialize Dictionary for single tile
    single_tile_dict = {}
    single_tile_dict["Tilename"] = tile_name

    # Convert tile to numpy array
    temp = tile  # DEEPZOOM_OBJECT.get_tile(DEEPZOOM_LEVEL - 1, (row, col))
    temp_rgb = temp.convert("RGB")

    temp = xp.array(temp_rgb)
    mean = xp.mean(temp)
    std = xp.std(temp)

    process_tile = False
    # Masking mode
    if mask is None:
        if mean < 230 and std > 15:
            process_tile = True
        else:
            single_tile_dict["Flag"] = 0
    else:
        process_tile = True

    if process_tile:
        # Separate stains
        DAB, H, E = color_deconvolution(temp)

        # Calculate pixel intensity
        (hist, hist_centers, zones, percentage, score, mask_count, img_analysis) = (
            evaluate_staining_intensities(DAB, antigen_profile, mask)
        )

        # Save image as tif in passed directory if wanted.
        if save_img:
            if not DAB_TILE_DIR:
                raise ValueError(
                    "Target directory must be specified if save_img is True"
                )
            img = Image.fromarray(to_numpy(DAB))
            DAB_TILE_DIR = f"{DAB_TILE_DIR}/{tile_name}.tif"
            # print(DAB_TILE_DIR)
            img.save(DAB_TILE_DIR)

        # Complete dictionary
        single_tile_dict["Histogram"] = hist
        single_tile_dict["Hist_centers"] = hist_centers
        single_tile_dict["Zones"] = zones
        single_tile_dict["Percentage"] = percentage
        single_tile_dict["Score"] = score
        single_tile_dict["Mask Count"] = mask_count
        single_tile_dict["Image Array"] = img_analysis
        single_tile_dict["Flag"] = 1

    return single_tile_dict


def color_deconvolution(
    ihc_rgb,
    hematoxylin=False,
    eosin=False,
):
    """
    Separates individual stains (Hematoxylin, Eosin, DAB) from an IHC image and returns an image for each stain.

    Args:
        ihc_rgb (Image): IHC image in RGB format.
        hematoxylin (bool): If True, returns Hematoxylin image as well. Defaults to False.
        eosin (bool): If True, returns Eosin image. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - ihc_d (Image): DAB (3',3'-Diaminobenzidine) stain of the image.
            - ihc_h (Image): Hematoxylin stain of the image if hematoxylin=True, otherwise None.
            - ihc_e (Image): Eosin stain of the image if eosin=True, otherwise None.
    """
    # Convert RGB image to HED using prebuilt skimage method
    # Convert to numpy array if transformation
    ihc_hed = rgb2hed(to_numpy(ihc_rgb))
    ihc_hed = xp.array(ihc_hed)
    # Create RGB image for each seperate stain
    null = xp.zeros_like(ihc_hed[:, :, 0])
    # Separate Hematoxylin stain
    ihc_h = (
        img_as_ubyte(
            hed2rgb(
                to_numpy(xp.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
        )
        if hematoxylin
        else None
    )
    # Separate Eosin stain
    ihc_e = (
        img_as_ubyte(
            hed2rgb(
                to_numpy(xp.stack((null, ihc_hed[:, :, 1], null), axis=-1)))
        )
        if eosin
        else None
    )
    # Separate DAB stain
    ihc_d = img_as_ubyte(
        hed2rgb(to_numpy(xp.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))
    )

    return ihc_d, ihc_h, ihc_e


def evaluate_staining_intensities(image, antigen_profile, tumor_mask=None):
    """
    Calculates pixel intensity of each pixel in the input image and separates them into 5 different zones based on
    their intensity. The image is converted to grayscale format, resulting in a distribution of intensity values
    between 0-255. Intensities above 235 are predominantly background or fatty tissues and do not contribute to
    pathological scoring. Thresholds for high-positive, medium-positive, low-positive and negative pixels are defined
    by the passed antigen profile. If 'masking_mode' is 'pixel-level', pixels with an intensity value of 255 will be
    excluded as they mark non-mask areas.
    After calculating pixel intensities this function calculates percentage contribution of each of the zones with
    respect to the mask tissue in the tile, as well as the a pathology score.

    Args:
        image (xp.ndarray): Input image.
        antigen_profile (dict): Dictionary with threshold values.
        tumor_mask (xp.ndarray). Boolean tumor mask for pixel-level, None for tile-level masking.

    Returns:
        tuple: A tuple containing:
            - hist (ndarray): Histogram of the image.
            - hist_centers (ndarray): Centers of histogram bins.
            - zones (xp.ndarray): Number of pixels in each intensity zone with respect to the tumor mask.
            - percentage (xp.ndarray): Percentage of pixels in each intensity zone with respect to the tumor mask.
            - score (str): Overall score of the tile.
            - maskcount (int): Tissue count for tile-level masking. Count of masked pixels for pixel-level.
            - img_analysis (xp.ndarray): Array of pixel values for multi-antigen evaluation.
    """

    # Conversion to gray-scale-ubyte image
    gray_scale_image = rgb2gray(image)
    gray_scale_ubyte = img_as_ubyte(gray_scale_image)

    # Calculate histogram
    hist, hist_centers = histogram(gray_scale_image)

    # Convert to xp array for processing
    gray_scale_ubyte = xp.array(gray_scale_ubyte)
    w, h = gray_scale_ubyte.shape

    # Initilize arrays for analysis
    img_analysis = xp.full((w, h), 255, dtype="float32")

    zones = xp.zeros(5, dtype=xp.int32)

    # Get thresholds from antigen_profile
    high_thresh = antigen_profile["high_positive_threshold"]
    medium_thresh = antigen_profile["medium_positive_threshold"]
    low_thresh = antigen_profile["low_positive_threshold"]

    # Define intensity masks
    high_positive_mask = gray_scale_ubyte < high_thresh
    positive_mask = (gray_scale_ubyte >= high_thresh) & (
        gray_scale_ubyte < medium_thresh
    )
    low_positive_mask = (gray_scale_ubyte >= medium_thresh) & (
        gray_scale_ubyte < low_thresh
    )
    negative_mask = (gray_scale_ubyte >= low_thresh) & (gray_scale_ubyte < 235)
    background_mask = gray_scale_ubyte >= 235

    if tumor_mask is not None:
        tumor_mask = xp.array(tumor_mask, dtype=bool)  # ensure boolean
        high_positive_mask &= tumor_mask
        positive_mask &= tumor_mask
        low_positive_mask &= tumor_mask
        negative_mask &= tumor_mask
        background_mask &= tumor_mask

    # Update img_analysis with pixel values based on intensity masks also containing background
    img_analysis[high_positive_mask] = gray_scale_ubyte[high_positive_mask]
    img_analysis[positive_mask] = gray_scale_ubyte[positive_mask]
    img_analysis[low_positive_mask] = gray_scale_ubyte[low_positive_mask]
    img_analysis[negative_mask] = gray_scale_ubyte[negative_mask]
    img_analysis[background_mask] = gray_scale_ubyte[background_mask]

    if tumor_mask is not None:
        img_analysis[~tumor_mask] = xp.nan
    # Calculate zones except non-mask
    zones[0] = xp.sum(high_positive_mask)
    zones[1] = xp.sum(positive_mask)
    zones[2] = xp.sum(low_positive_mask)
    zones[3] = xp.sum(negative_mask)
    zones[4] = xp.sum(background_mask)

    tissue_count = xp.sum(zones[:4])
    if tumor_mask is not None:
        mask_count = xp.sum(tumor_mask)
    else:
        mask_count = tissue_count
    # Calculate pixel count and percentage

    if xp.sum(zones[:4]) == 0:
        percentage = xp.zeros(5, dtype=xp.int32)
        percentage[4] = 100
        score = "Background"
    else:
        percentage, score = calculate_percentage_and_score(zones)

    return (
        hist,
        hist_centers,
        zones,
        percentage,
        score,
        int(mask_count),
        to_numpy(img_analysis),
    )


def calculate_percentage_and_score(zones):
    """
    Calculates the percentage of pixels in each zone relative to the tissue count (Positive tissues) and total mask
    count (Background) and computes a score for each zone. If more than 66.6% of the total pixels are attributed to
    a single zone, that zone's score is assigned. Else, the score for each zone is calculated using this formula:

    .. math::

        \\text{Score} = \\frac{(\\text{number of pixels in zone} \\times \\text{weight of zone})}{\\text{total
        pixels in image}}

    with weights 4 for the high positive zone, 3 for the positive zone, 2 for the low positive zone, 1 for the negative
    zone, and 0 for the background. The final score is the maximum score among all zones.

    Args:
        zones (xp.ndarray): Array containing amount of pixels from each zone

    Returns:
        tuple: A tuple containing:
            - percentage (xp.ndarray): Array containing the percentage of pixels in each zone.
            - score (str): Name of the zone if it exceeds 66.6%, otherwise the name of the zone with the highest score.

    Raises:
        ValueError: If all zones have zero pixels.
    """

    if xp.sum(zones) == 0:
        raise ValueError("All zones have zero pixels")

    tissue_count = xp.sum(zones[:4])
    total_pixels = xp.sum(zones)

    # Calculate percentage of pixels in each zone
    percentage_tissue = (zones[:4] / tissue_count) * 100
    percentage_background = (zones[4:] / total_pixels) * 100
    percentage = xp.concatenate([percentage_tissue, percentage_background])
    zone_names = [
        "High Positive",
        "Medium Positive",
        "Low Positive",
        "Negative",
        "Background",
    ]

    # Check if any zone exceeds 66.6% of the total pixels
    if xp.any(percentage > 66.6):
        max_score_index = int(xp.argmax(percentage))
        return percentage, zone_names[max_score_index]

    # Else calculate wheighted score for each zone
    weights = xp.array([4, 3, 2, 1, 0])  # Weights for each zone
    scores = (zones * weights) / tissue_count
    max_score_index = int(xp.argmax(scores[:4]))  # ignore background zone

    return percentage, zone_names[max_score_index]


def mask_tile(tile, mask):
    """
    Masks the tile with the given mask. The mask is a binary image with the same dimensions as the tile. The function
    returns the masked tile as an Image, containing the tile where the mask is positive and white where it is negative.

    Args:
        tile (Image): Tile to be masked
        mask (Image): Mask to be applied to the tile

    Returns:
        Image: Masked tile
    """
    # Convert tile to numpy array
    tile_np = np.array(tile)
    mask_np = np.array(mask)

    if len(mask_np.shape) == 3:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
    # Resize mask to tile size if needed
    if mask_np.shape != tile_np.shape[:2]:
        mask_np = cv2.resize(
            mask_np,
            (tile_np.shape[1], tile_np.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)

    tumor_mask = binary_mask == 0

    binary_mask_inv = cv2.bitwise_not(binary_mask)
    binary_mask_inv_3ch = cv2.merge(
        (binary_mask_inv, binary_mask_inv, binary_mask_inv))

    masked_tile = cv2.bitwise_and(tile_np, binary_mask_inv_3ch)
    white_bg = np.ones_like(tile_np) * 255
    masked_tile = np.where(binary_mask_inv_3ch == 0, white_bg, masked_tile)

    return Image.fromarray(masked_tile.astype(np.uint8)), tumor_mask


def init_worker(slide_path, mask_path, level, dab_tile_dir, save_img, antigen_profile):
    """Runs once per worker: opens slide (and mask) locally."""
    osr = openslide.OpenSlide(slide_path)
    _state["osr"] = osr  # keep reference
    _state["tiles"] = DeepZoomGenerator(osr, tile_size=1024, overlap=0, limit_bounds=True)
    _state["mask_tiles"] = None
    if mask_path is not None:
        mosr = openslide.OpenSlide(mask_path)
        _state["mosr"] = mosr
        _state["mask_tiles"] = DeepZoomGenerator(mosr, tile_size=1024, overlap=0, limit_bounds=True)
    _state.update(level=level, dir=dab_tile_dir, save_img=save_img, profile=antigen_profile)


def quantify_tile_at(xy):
    t0 = time.perf_counter()  # TODO remove
    x, y = xy
    level = _state["level"]
    tile = _state["tiles"].get_tile(level, (x, y))
    if _state["mask_tiles"] is not None:
        tile_data = mask_tile(tile, _state["mask_tiles"].get_tile(level, (x, y)))
    else:
        tile_data = (tile, None)
    t1 = time.perf_counter()

    result = quantify_tile(
        (x, y, tile_data, _state["dir"], _state["save_img"], _state["profile"])
    )
    t2 = time.perf_counter()

    result["_t_read"] = t1 - t0      # tile reading (+ masking in pixel-level mode)
    result["_t_compute"] = t2 - t1   # quantify_tile: transfers, deconvolution, histograms
    return result
