# Standard Library
import concurrent.futures
import logging
import os
import pickle
from time import time

# Third Party
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from pyvips import BandFormat
from pyvips import Image as VipsImage
from tqdm import tqdm

# CuBATS
import cubats.logging_config as log_config
from cubats.config import xp
from cubats.slide_collection.tile_quantification import (mask_tile,
                                                         quantify_tile)


class Slide(object):
    """Slide Class.

    `Slide` class instatiates a slide object containing all relevant information and results for a single slide. All
     slide specific operations that can be performed on a single slide rather than on a collection of slides are
     implemented in this class. This includes quantification of staining intensities and reconstruction of a slide. The
     class is initialized with the name of the slide, the path to the slide file, as well as information on whether the
     slide is a mask or reference slide. The class contains a dictionary of detailed quantification results for each
     tile, as well as a dictionary of summarized quantification results for the entire slide. The slide object also
     contains information on the OpenSlide object, the tiles, the level count, the level dimensions, as well as the
     tile count. The slide object also contains a directory to save the tiles after color deconvolution, which is
     necessary for reconstruction of the slide later on.

    Attributes:
        name (str): The name of the slide.

        openslide_object (openslide.OpenSlide): The OpenSlide Object of the slide.

        tiles (openslide.deepzoom.DeepZoomGenerator): DeepZoom Generator containing the tiles of the slide.

        level_count (int): The number of DeepZoom levels of the slide.

        level_dimensions (list):  List of tuples (pixels_x, pixels_y) for each Deep Zoom level.

        tile_count (int): The number of tiles in the slide.

        masked_tiles (list): List containing the tiles after intersection of the slide and mask.

        dab_tile_dir (str): Directory to save the tiles after color deconvolution is applied. Necessary for
            reconstruction of the slide. If save_img is False no tiles are saved and this attribute is None.

        is_mask (bool): Whether the slide is the slide_collections mask slide.

        is_reference (bool): Whether the slide is the slide_collections reference slide.

        antigen_profile (dict, optional): Dictionary containing antigen-specific quantification thresholds
            (high-positive, medium-positive, low-postitive). If no profile is provided, default thresholds are used.

        detailed_quantification_results (dict): Dictionary containing detailed quantification results for each tile of
            the slide. The dictionary is structured as follows:

            - key (int): Index of the tile.
            - value (dict): Dictionary containing the following:

                - Flag (int): Flag indicating whether the tile was processed (1) or not (0).
                - Histogram (array): Array containing the histogram of the tile.
                - Hist_centers (array): Array containing the centers of the histogram bins.
                - Zones (array): Array containing the number of pixels in each zone sorted by index according to the
                  following attribution High Positive, Positive, Low Positive, Negative, Background).
                - Score (str): Score of the tile based on the zones.

        quantification_summary (dict): Dictionary containing the summarized quantification results for the slide:

            - Name (str): Slide name
            - Coverage (%) (float): Percentage of all positively quantified pixels in the Slide. This includes all
              high-postive, medium-positive, and low-positive stained pixels.
            - High Positive (%) (float): Percentage of highly positive stained pixels in the Slide.
            - Medium Positive (%) (float): Percentage of medium positive stained pixels in the Slide.
            - Low Positive  (%) (float): Percentage of low positive stained pixels in the Slide.
            - Negative  (%) (float): Percentage of negatively stained pixels in the Slide.
            - Total Tissue  (%) (float): Percentage of tissue in the Slide.
            - Background / No Tissue (%) (float): Percentage of Background pixels in the Slide.
            - H-Score (int): H-Score of the slide.
            - Score (str): Pathological score of the slide: High Positive, Medium Positive, Low Positive, Negative,
              Background.
            - Total Processed Tiles (%): Percentage of processed tiles of the slide.
            - Error (%) (float): Percentage of skipped tiles because they did not contain tissue.
            - Thresholds: Antigen-specific thresholds used during quantification.

        properties (dict): Dictionary containing relevant slide properties including: `name`, `is_reference`, `is_mask`,
            `openslide_object`, `tiles`, `level_count`, `level_dimensions`, and `tile_count`.
    """

    def __init__(self, name, path, is_mask=False, is_reference=False):
        """
        Initialize a Slide object.

        Args:
            name (str): The name of the slide.
            path (str): The path to the slide file.
            is_mask (bool, optional): Whether the slide is a mask. Defaults to False.
            is_reference (bool, optional): Whether the slide is a reference slide. Defaults to False.
        """
        # Initialize logger
        logging.config.dictConfig(log_config.LOGGING)
        self.logger = logging.getLogger(__name__)

        self.name = name
        self.orig_path = path
        self.registered_path = None
        self.openslide_object = openslide.OpenSlide(path)
        self.tiles = DeepZoomGenerator(
            self.openslide_object, tile_size=1024, overlap=0, limit_bounds=True
        )
        self.level_count = self.tiles.level_count
        self.level_dimensions = self.tiles.level_dimensions
        self.tile_count = self.tiles.tile_count
        self.masked_tiles = None

        self.dab_tile_dir = None

        self.is_mask = is_mask
        self.is_reference = is_reference

        self.detailed_quantification_results = {}
        self.quantification_summary = {}

        if not is_mask and not is_reference:
            self.antigen_profile = {
                "Name": "default",
                "low_positive_threshold": 181,
                "medium_positive_threshold": 121,
                "high_positive_threshold": 61,
            }
        else:
            self.antigen_profile = None

        self.properties = {
            "name": self.name,
            "reference": self.is_reference,
            "mask": self.is_mask,
            "openslide_object": self.openslide_object,
            "tiles": self.tiles,
            "level_count": self.level_count,
            "level_dimensions": self.level_dimensions,
            "tile_count": self.tile_count,
        }
        self.logger.debug(f"Slide {self.name} initialized.")

    def quantify_slide(
        self,
        mask_coordinates,
        save_dir,
        save_img=False,
        img_dir=None,
        mask=None,
    ):
        """Quantifies staining intensities for masked tiles of this slide.

        This function uses multiprocessing to quantify staining intensities of masked tiles for the slide.
        Each tile undergoes color deconvolution followed by staining intensity quantification based on the IHC
        Profiler's algorithm. If `save_img` is True, tiles are saved in `img_dir` after deconvolution for later
        reconstruction. After color deconvolution, each tile is processed as a grayscale image, and each pixel's
        staining intensity (0-255) is quantified based on the thresholds defined in `self.antigen_profile`. If no
        specific antigen_profile was provided the default profile will be used. Results are stored in
        `self.detailed_quantification_results` and summarized in `self.quantification_summary`. Both are saved as
        PICKLE files in `save_dir`.

        Args:
            mask_coordinates (list): List of xy-coordinates from the maskslide where the mask is positive.

            save_dir (str): Directory to save the results. Usually the slides pickle directory.

            save_img (bool, optional): Whether to save the tiles after color deconvolution. Defaults to False.
                Necessary for reconstruction of the slide.

            img_dir (str, optional): Directory to save the tiles. Must be provided if tiles shall be saved. Defaults to
                None.

            mask (openslide.deepzoom.DeepZoomGenerator, optional): DeepZoomGenerator containing the detailed
                mask. Defaults to None. Provides a more detailed mask for the quantification of the slide, however,
                might result in larger inaccuracies for WSI with low congruence.

        """
        self.logger.debug(
            f"Quantifying slide: {self.name}, antigen_profile: "
            f"{self.antigen_profile['Name'] if self.antigen_profile else 'None'}, "
            f"save_img: {save_img}, masking_mode: {'pixel-level' if mask is not None else 'tile-level'}"
        )
        if self.is_mask:
            self.logger.error("Cannot quantify mask slide.")
            raise ValueError("Cannot quantify mask slide.")
        elif self.is_reference:
            self.logger.error("Cannot quantify reference slide.")
            raise ValueError("Cannot quantify reference slide.")

        # masking_mode = "pixel-level" if mask is not None else "tile-level"

        # Create directory to save tiles if save_img is True
        if save_img:
            if img_dir is None:
                self.logger.error(
                    "img_dir must be provided if save_img is True.")
                raise ValueError(
                    "img_dir must be provided if save_img is True.")
            self.dab_tile_dir = img_dir
            os.makedirs(self.dab_tile_dir, exist_ok=True)

        start_time_preprocessing = time()
        # Creates an iterable containing xy-Tuples for each tile, DeepZoomGenerator, and directory.
        iterable = [
            (
                x,
                y,
                (
                    mask_tile(
                        self.tiles.get_tile(self.level_count - 1, (x, y)),
                        mask.get_tile(self.level_count - 1, (x, y)),
                    )
                    if mask is not None
                    else (self.tiles.get_tile(self.level_count - 1, (x, y)), None)
                ),
                self.dab_tile_dir,
                save_img,
                self.antigen_profile,
                # masking_mode,
            )
            for x, y in tqdm(
                mask_coordinates,
                desc="Pre-processing slide: " + self.name,
                total=len(mask_coordinates),
            )
        ]
        end_time_preprocessing = time()
        if end_time_preprocessing - start_time_preprocessing >= 60:
            self.logger.info(
                f"Finished pre-processing slide: {self.name} in \
                    {round((end_time_preprocessing - start_time_preprocessing) / 60, 2)} minutes."
            )
        else:
            self.logger.info(
                f"Finished pre-processing slide: {self.name} in \
                    {round((end_time_preprocessing - start_time_preprocessing), 2)} seconds."
            )

        start_time_quantification = time()
        max_workers = os.cpu_count() - 1
        # k = 4
        # chunksize = max(1, len(iterable) // (max_workers * k))
        # Multiprocessing using concurrent.futures, gathering results and adding them to dictionary in linear manner.
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as exe:
            results = tqdm(
                exe.map(quantify_tile, iterable),
                total=len(iterable),
                desc="Processing slide: " + self.name,
            )
            for idx, res in enumerate(results):
                # if result is not None:
                self.detailed_quantification_results[idx] = res
        end_time_quantification = time()
        if end_time_quantification - start_time_quantification >= 60:
            self.logger.info(
                f"Finished quantifying slide: {self.name} in \
                    {round((end_time_quantification - start_time_quantification) / 60, 2)} minutes."
            )
        else:
            self.logger.info(
                f"Finished quantifying slide: {self.name} in \
                    {round((end_time_quantification - start_time_quantification), 2)} seconds."
            )

        # Retrieve Quantification results and save to disk
        self.summarize_quantification_results()

        # Save dictionary as pickle
        start_time_save = time()
        f_out = os.path.join(save_dir, f"{self.name}_processing_info.pickle")
        self.logger.info(
            f"Saving quantification results for {self.name} to {f_out}")
        with open(f_out, "wb") as f:
            pickle.dump(
                self.detailed_quantification_results,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        end_time_save = time()
        self.logger.debug(
            f"Saved quantification results for {self.name} to {f_out} in \
                {round((end_time_save - start_time_save) / 60, 2)} minutes."
        )
        self.logger.info(f"Finished processing slide: {self.name}")

    def summarize_quantification_results(self):
        """
        Summarizes quantification results.

        Summarizes quantification results for a given slide and appends them to `self.quantification_summary`. This
        includes the total number of pixels in each zone, the percentage of pixels in each zone, and a score for each
        zone.

        The summary contains the following keys:
            - Slide (str): Name of the slide.
            - Coverage (float): Tumor coverage of the antigen in the slide.
            - High Positive (float): Percentage of pixels in the high positive zone.
            - Positive (float): Percentage of pixels in the positive zone.
            - Low Positive (float): Percentage of pixels in the low positive zone.
            - Negative (float): Percentage of pixels in the negative zone.
            - Total Tissue (float): Total amount of tissue in the slide.
            - Background (float): Percentage of pixels in the white space background or fatty tissues.
            - H-Score (float): H-score calculation based on positive pixels.
            - Score (str): Overall score of the slide based on the zones. However, the score for the entire slide
              may be misleading since much negative tissue may lead to a negative score even though the slide may
              contain a lot of positive tissue as well. Therefore, the score for the entire slide should be
              interpreted with caution.
            - Total Processed Tiles (float): Percentage of total processed tiles.
            - Error (float): Percentage of tiles that were not processed as they did not contain sufficient tissue.

        Raises:
            ValueError: If the slide is a mask slide or a reference slide.
        """
        start_time_summarize = time()
        self.logger.info(
            f"Summarizing quantification results for slide: {self.name}")
        if self.is_mask:
            self.logger.error(
                "Cannot summarize quantification results for mask slide.")
            raise ValueError(
                "Cannot summarize quantification results for mask slide.")
        elif self.is_reference:
            self.logger.error(
                "Cannot summarize quantification results for reference slide."
            )
            raise ValueError(
                "Cannot summarize quantification results for reference slide."
            )

        # Init variables
        zone_names = [
            "High Positive",
            "Medium Positive",
            "Low Positive",
            "Negative",
            "Background",
        ]
        wheights = xp.array([4, 3, 2, 1, 0])
        sum_zones = xp.zeros((5,), dtype=xp.float32)
        sum_mask_count = 0
        processed_tiles = 0
        error_tiles = 0

        # Iterate through tiles and aggregate results
        for tile_result in self.detailed_quantification_results.values():
            if tile_result["Flag"] == 1:
                processed_tiles += 1
                sum_zones += xp.array(tile_result["Zones"], dtype=xp.float32)
                sum_mask_count += tile_result["Mask Count"]
            else:
                error_tiles += 1

        if processed_tiles == 0:
            self.logger.warning(
                f"No tiles were successfully processed for slide: {self.name}. "
                "Skipping quantification summary calculations."
            )
            self.quantification_summary = {
                "Name": self.name,
                "Coverage (%)": 0.0,
                "High Positive (%)": 0.0,
                "Medium Positive (%)": 0.0,
                "Low Positive (%)": 0.0,
                "Negative (%)": 0.0,
                "Total Tissue (%)": 0.0,
                "Background / No Tissue (%)": 0.0,
                "Mask Area (%)": 0.0,
                "Non-mask Area (%)": 0.0,
                "H-Score": 0.0,
                "Score": "Background",
                "Total Processed Tiles (%)": 0.0,
                "Error (%)": 100.0,
                "Thresholds": [
                    self.antigen_profile["high_positive_threshold"],
                    self.antigen_profile["medium_positive_threshold"],
                    self.antigen_profile["low_positive_threshold"],
                    235,
                ],
            }
            return  # Exit early

        # Calculate percentages and scores
        total_pixels_in_mask = sum_mask_count
        tissue_count = xp.sum(sum_zones[:4])
        total_pixels = processed_tiles * 1048576
        total_non_mask_pixels = total_pixels - total_pixels_in_mask

        # Percentage for intensity zones based on tissue count
        percentage_high_pos = (sum_zones[0] / total_pixels_in_mask) * 100
        percentage_med_pos = (sum_zones[1] / total_pixels_in_mask) * 100
        percentage_low_pos = (sum_zones[2] / total_pixels_in_mask) * 100
        percentage_neg = (sum_zones[3] / total_pixels_in_mask) * 100

        # percentages_tissue = (sum_zones[:4] / total_non_mask_pixels) * 100

        # Percentage for background based on total pixels
        percentage_background = (sum_zones[4] / total_pixels_in_mask) * 100

        # Coverage percentage of positively stained tissue (high+medium+low)
        coverage_pixels = sum_zones[0] + sum_zones[1] + sum_zones[2]
        perc_coverage = (coverage_pixels / total_pixels_in_mask) * 100

        # Total percentage of tissue
        perc_tissue = (tissue_count / total_pixels_in_mask) * 100

        perc_mask = (total_pixels_in_mask / total_pixels) * 100
        perc_non_mask = (total_non_mask_pixels / total_pixels) * 100

        # Calculate H-Score
        percentages_tissue = xp.array(
            [
                percentage_high_pos,
                percentage_med_pos,
                percentage_low_pos,
                percentage_neg,
                percentage_background,
            ],
            dtype=xp.float32,
        )
        h_score = xp.sum(
            percentages_tissue[:3] * xp.array([3, 2, 1]), dtype=xp.float32)

        # Determine slide score (exclude background)
        if xp.any(percentages_tissue > 66.6):
            max_score_index = int(xp.argmax(percentages_tissue))
        else:
            scores = (sum_zones * wheights) / processed_tiles
            max_score_index = int(xp.argmax(scores[:4]))

        slide_score = zone_names[max_score_index]

        # Calculate percentage of processed tiles and error
        perc_processed_tiles = (
            processed_tiles / len(self.detailed_quantification_results)
        ) * 100
        perc_error = (
            error_tiles / len(self.detailed_quantification_results)) * 100

        # Update the dictionary
        self.quantification_summary = {
            "Name": self.name,
            "Coverage (%)": round(float(perc_coverage), 4),
            "High Positive (%)": round(float(percentage_high_pos), 4),
            "Medium Positive (%)": round(float(percentage_med_pos), 4),
            "Low Positive (%)": round(float(percentage_low_pos), 4),
            "Negative (%)": round(float(percentage_neg), 4),
            "Total Tissue (%)": round(float(perc_tissue), 4),
            "Background / No Tissue (%)": round(float(percentage_background), 4),
            "Mask Area (%)": round(float(perc_mask), 4),
            "Non-mask Area (%)": round(float(perc_non_mask), 4),
            "H-Score": round(float(h_score), 2),
            "Score": slide_score,
            "Total Processed Tiles (%)": round(float(perc_processed_tiles), 4),
            "Error (%)": round(float(perc_error), 4),
            "Thresholds": [
                self.antigen_profile["high_positive_threshold"],
                self.antigen_profile["medium_positive_threshold"],
                self.antigen_profile["low_positive_threshold"],
                235,
            ],
        }
        end_time_summarize = time()
        self.logger.debug(
            f"Finished summarizing quantification results for slide: {self.name} in \
                {round((end_time_summarize - start_time_summarize) / 60, 2)} minutes."
        )

    def reconstruct_slide(self, in_path, out_path):
        """
        Reconstructs a slide into a Whole Slide Image (WSI) based on saved tiles. This is only possible if tiles have
        been saved during processing. The WSI is then saved as .tif in the specified `out_path`.

        Args:
            in_path (str): Path to saved tiles
            out_path (str): Path where to save the reconstructed slide.

        """
        start_time = time()
        # Check paths
        if not os.path.isdir(in_path):
            self.logger.error(f"Input path {in_path} does not exist.")
            raise ValueError(f"Input path {in_path} does not exist.")
        # Check if in_path contains .tif files
        tif_files = [f for f in os.listdir(in_path) if f.endswith(".tif")]
        if not tif_files:
            self.logger.error(f"No .tif files found in input path {in_path}.")
            raise ValueError(f"No .tif files found in input path {in_path}.")

        # Ensure output directory exists
        os.makedirs(out_path, exist_ok=True)

        # Init variables
        counter = 0
        cols, rows = self.tiles.level_tiles[self.level_count - 1]
        row_array = []

        # append tiles for each column and row. Previously not processed tiles are replaced by white tiles.
        for row in tqdm(range(rows), desc="Reconstructing slide: " + self.name):
            column_array = []
            for col in range(cols):
                tile_name = str(col) + "_" + str(row)
                file = os.path.join(in_path, tile_name + ".tif")
                if os.path.exists(file):
                    img = Image.open(file)
                    counter += 1
                else:
                    img = Image.new("RGB", (1024, 1024), (192, 192, 192))

                column_array.append(img)

            segmented_row = np.concatenate(column_array, axis=1)
            row_array.append(segmented_row)

        # Create WSI and save as pyramidal TIF in self.reconstruct_dir
        logging.getLogger("pyvips").setLevel(logging.WARNING)
        segmented_wsi = np.concatenate(row_array, axis=0)
        segmented_wsi = VipsImage.new_from_array(
            segmented_wsi).cast(BandFormat.INT)
        end_time = time()
        self.logger.info(
            f"Finished reconstructing slide: {self.name} in {round((end_time - start_time) / 60, 2)} minutes."
        )

        start_time_save = time()
        out = os.path.join(out_path, self.name + "_reconst.tif")
        self.logger.info(f"Saving reconstructed slide to {out}")
        segmented_wsi.crop(
            0,
            0,
            self.openslide_object.dimensions[0],
            self.openslide_object.dimensions[1],
        ).tiffsave(
            out,
            tile=True,
            compression="jpeg",
            bigtiff=True,
            pyramid=True,
            tile_width=256,
            tile_height=256,
        )
        end_time_save = time()
        self.logger.debug(
            f"Saved reconstructed slide to {out} in {round((end_time_save - start_time_save) / 60, 2)} minutes."
        )

    def update_slide(self, new_path):
        """
        Updates the path of the slide and reinitializes the OpenSlide object.

        Args:
            new_path (str): The new path to the slide file.
        """
        self.logger.debug(f"Updating path for slide {self.name} to {new_path}")
        self.registered_path = new_path
        self.openslide_object = openslide.OpenSlide(new_path)
        self.tiles = DeepZoomGenerator(
            self.openslide_object, tile_size=1024, overlap=0, limit_bounds=True
        )
        self.level_count = self.tiles.level_count
        self.level_dimensions = self.tiles.level_dimensions
        self.tile_count = self.tiles.tile_count
        self.logger.debug(
            f"Slide {self.name} updated with new path: {new_path}")
