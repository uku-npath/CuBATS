# Standard Library
import concurrent.futures
import logging.config
import multiprocessing as mp
import os
import pickle
import re
from itertools import combinations
from time import time

# Third Party
import pandas as pd
from PIL import Image
from tqdm import tqdm

# CuBATS
import cubats.cutils as cutils
import cubats.logging_config as log_config
from cubats.config import get_backend_info, xp
from cubats.slide_collection.registration import (
    register_slides, register_slides_high_resolution,
    register_slides_with_reference)
from cubats.slide_collection.segmentation import run_tumor_segmentation
from cubats.slide_collection.slide import Slide
from cubats.slide_collection.tile_colocalization import (
    evaluate_antigen_pair_tile, evaluate_antigen_triplet_tile)

# Constants
# Destination directories
RESULT_DATA_DIR = "data"
REGISTRATION_DIR = "registration"  # registered_slides
TILES_DIR = "tiles"
COLOCALIZATION = "colocalization"
ORIGINAL_TILES_DIR = "original"
DAB_TILE_DIR = "dab"
E_TILE_DIR = "eosin"
H_TILE_DIR = "hematoxylin"
PICKLE_DIR = "pickle"
RECONSTRUCT_DIR = "reconstructed_slides"
SLIDE_COLLECTION_COLUMN_NAMES = [
    "Name",
    "Reference",
    "Mask",
    "Openslide Object",
    "Tiles",
    "Level Count",
    "Level Dimensions",
    "Tile Count",
]
SUPPORTED_IMAGE_FORMATS = [".tif", ".tiff", ".mrxs", ".svs"]
# Define column names and data types
QUANTIFICATION_RESULTS_COLUMN_NAMES = {
    "Name": str,
    "Coverage (%)": float,
    "High Positive (%)": float,
    "Medium Positive (%)": float,
    "Low Positive (%)": float,
    "Negative (%)": float,
    "Total Tissue (%)": float,
    "Background / No Tissue (%)": float,
    "Mask Area (%)": float,
    "Non-mask Area (%)": float,
    "H-Score": float,
    "Score": str,
    "Total Processed Tiles (%)": float,
    "Error (%)": float,
    "Thresholds": str,
}
DUAL_ANTIGEN_EXPRESSIONS_COLUMN_NAMES = {
    "Slide 1": str,
    "Slide 2": str,
    "Total Coverage (%)": float,
    "Total Overlap (%)": float,
    "Total Complement (%)": float,
    "High Positive Overlap (%)": float,
    "High Positive Complement (%)": float,
    "Medium Positive Overlap (%)": float,
    "Medium Positive Complement (%)": float,
    "Low Positive Overlap (%)": float,
    "Low Positive Complement (%)": float,
    "Negative Tissue (%)": float,
    "Total Tissue (%)": float,
    "Background / No Tissue (%)": float,
    "Mask Area (%)": float,
    "Non-mask Area (%)": float,
    "Total Processed Tiles (%)": float,
    "Total Error (%)": float,
    "Error1 (%)": float,
    "Error2 (%)": float,
    "Thresholds1": str,
    "Thresholds2": str,
}
TRIPLET_ANTIGEN_EXPRESSIONS_COLUMN_NAMES = {
    "Slide 1": str,
    "Slide 2": str,
    "Slide 3": str,
    "Total Coverage (%)": float,
    "Total Overlap (%)": float,
    "Total Complement (%)": float,
    "High Positive Overlap (%)": float,
    "High Positive Complement (%)": float,
    "Medium Positive Overlap (%)": float,
    "Medium Positive Complement (%)": float,
    "Low Positive Overlap (%)": float,
    "Low Positive Complement (%)": float,
    "Negative Tissue (%)": float,
    "Total Tissue (%)": float,
    "Background / No Tissue (%)": float,
    "Mask Area (%)": float,
    "Non-mask Area (%)": float,
    "Total Error (%)": float,
    "Error1 (%)": float,
    "Error2 (%)": float,
    "Thresholds1": str,
    "Thresholds2": str,
    "Thresholds3": str,
}
DEFAULT_TILE_SIZE = 1024


class SlideCollection(object):
    """Initializes a slide collection, stores slide info and performs slide processing.

    `SlideCollection` is a class that initializes a collection of slides, stores relevant processing information,
    and allows the execution of separate processing steps. Previously processed data can be reloaded.

    Attributes:
        name (str): Name of parent directory (i.e. name of tumorset).

        src_dir (str): Path to source directory containing the WSIs.

        dest_dir (str): Path to destination directory for results.

        data_dir (str): Path to data directory, a subdirectory of `dest_dir`. The directory will be initiaded upon class
            creation inside the `dest_dir`. Inside the data directory summaries of quantification results, dual antigen
            expression results and triplet antigen expression results are stored as .CSV file.
            The data_dir also contains the `pickle_dir`.

        pickle_dir (str): Path to pickle directory, a subdirectory of `data_dir`. Inside the pickle directory pickled
            copies of the slide_collection, quantification results and antigen analysis are stored which will be
            automatically reloaded if a slide collection is (re-)initialized with the same output_dir.

        tiles_dir (str): Path to tiles directory, a subdirectory of `dest_dir`. Inside the tiles directory the tile
            directories for each slide are stored.

        colocalization_dir (str): Path to colocalization directory, a subdirectory of `dest_dir`. Inside the
            colocalization directory results of dual and triplet overlap analyses are stored.

        reconstruct_dir (str): Path to reconstruct directory, a subdirectory of `dest_dir`. Inside the reconstruct
            directory reconstructed slides are stored.

        collection_list (list of Slide): list containing all slide objects.

        collection_info_df (Dataframe): Dataframe containing relevant information on all the slides. The colums are:

            - Name (str): Name of slide.
            - Reference (bool): True if slide is reference slide.
            - Mask (bool): True if slide is mask slide.
            - OpenSlide Object (OpenSlide): OpenSlide object of the slide.
            - Tiles (DeepZoomGenerator): DeepZoom tiles of the slide.
            - Level Count (int): Number of Deep Zoom levels in the image.
            - Level Dimensions (list): List of tuples (pixels_x, pixels_y) for each Deep Zoom level.
            - Tile Count (int): Number of total tiles in the image.

        mask (Slide): Mask slide of the collection. Is set during initialization.

        mask_coordinates (list): List containing the tile coordinates for tiles that are covered by the mask
            Coordinates are tuples (column, row).

        quantification_results (Dataframe): Dataframe containing the quantification results for all processed slides.
            The columns are:

            - Name (str): Name of the slide.

            - Coverage (%) (float): Positively stained pixels in the slide, characterizing the tumor coverage.

            - High Positive (%) (float): Percentage of highly positive stained pixels in the slide.

            - Medium Positive (%) (float): Percentage of medium positive stained pixels in the slide.

            - Low Positive (%) (float): Percentage of low positive stained pixels in the slide.

            - Negative (%) (float): Percentage negatively stained pixels in the slide.

            - Total Tissue (%) (float): Total amount of tissue in the slide.

            - Background / No Tissue (%) (float): Percentage of background and non-tissue regions in the slide.

            - Mask Area (%) (float): Area of the slide covered by the tumor mask.

            - Non-mask Area (%) (float): Area of the slide not covered by the tumor mask.

            - H-Score (float): Established pathological score calculated based on the distribution of staining
              intensities in the slide.

            - Score (str): Additional overall score of the slide calculated from the average of scores for all tiles.
              However, this score may be misleading, as it is an average over the entire slide.

            - Total Processed Tiles (%) (float): Percentage of processed tiles.

            - Error (%) (float): Percentage of tiles that were not processed due to insufficient tissue coverage.
              Only the case for tile-level masking.

            - Thresholds (list): List containing the thresholds applied during quantification.

        dual_antigen_expressions (Dataframe): Dataframe containing a summary of the dual antigen expression results
            for all processed analyses:

            - Slide 1 (str): Name of the first slide.

            - Slide 2 (str): Name of the second slide.

            - Total Coverage (%) (float): Percentage of combined coverage in two slides.

            - Total Overlap (%) (float: Percentage of overlapping antigen expressions in two slides.

            - Total Complement (%) (float): Percentage of complementary antigen expressions in two slides.

            - High Positive Overlap (%) (float): Percentage of highly positive overlapping antigen expressions in
              two slides.

            - High Positive Complement (%) (float): Percentage of highly positive complementary antigen expressions in
              two slides.

            - Medium Positive Overlap (%) (float): Percentage of medium positive overlapping antigen expressions in
              two slides.

            - Medium Positive Complement (%) (float): Percentage of medium positive complementary antigen expressions
              in two slides.

            - Low Positive Overlap (%) (float): Percentage of low positive overlapping antigen expressions in
              two slides.

            - Low Positive Complement (%) (float): Percentage of low positive complementary antigen expressions in
              two slides.

            - Negative Tissue (%) (float): Percentage of negative tissue in the two slides.

            - Total Tissue (%) (float): Percentage of total tissue in the two slides.

            - Background / No Tissue (%) (float): Percentage of background and non-tissue regions in the two slides.

            - Total Processed Tiles (%) (float): Percentage of processed tiles.

            - Total Error (%) (float): Percentage of tiles that were not processed due to insufficient tissue coverage.

            - Error1 (%) (float): Percentage of tiles where neither tile contained tissue.

            - Error2 (%) (float): Percentage of tiles where tiles could not be analyzed due to incorrect tile shapes.

            - Thresholds1 (str): Antigen thresholds for slide 1.

            - Thresholds2 (str): Antigen thresholds for slide 2

        triplet_antigen_results (Dataframe): Dataframe containing a summary of the triplet antigen expression results
            for all processed analyses:

            - Slide 1 (str): Name of the first slide.

            - Slide 2 (str): Name of the second slide.

            - Slide 3 (str): Name of the third slide.

            - Total Coverage (%) (float): Percentage of combined coverage in three slides.

            - Total Overlap (%) (float: Percentage of overlapping antigen expressions in three slides.

            - Total Complement (%) (float): Percentage of complementary antigen expressions in three slides.

            - High Positive Overlap (%) (float): Percentage of highly positive overlapping antigen expressions in
              three slides.

            - High Positive Complement (%) (float): Percentage of highly positive complementary antigen expressions in
              three slides.

            - Medium Positive Overlap (%) (float): Percentage of medium positive overlapping antigen expressions in
              three slides.

            - Medium Positive Complement (%) (float): Percentage of medium positive complementary antigen expressions
              in three slides.

            - Low Positive Overlap (%) (float): Percentage of low positive overlapping antigen expressions in
              three slides.

            - Low Positive Complement (%) (float): Percentage of low positive complementary antigen expressions in
              three slides.

            - Negative Tissue (%) (float): Percentage of negative tissue in the three slides.

            - Total Tissue (%) (float): Percentage of total tissue in the three slides.

            - Background / No Tissue (%) (float): Percentage of background and non-tissue regions in the three slides.

            - Total Processed Tiles (%) (float): Percentage of processed tiles.

            - Total Error (%) (float): Percentage of tiles that were not processed due to insufficient tissue coverage.

            - Error1 (%) (float): Percentage of tiles where neither tile contained tissue.

            - Error2 (%) (float): Percentage of tiles where tiles could not be analyzed due to incorrect tile shapes.

            - Thresholds1 (str): Antigen thresholds for slide 1.

            - Thresholds2 (str): Antigen thresholds for slide 2

            - Thresholds3 (str): Antigen thresholds for slide 3
    """

    def __init__(
        self,
        collection_name,
        src_dir,
        dest_dir,
        ref_slide=None,
        path_antigen_profiles=None,
    ):
        """Initializes the class. The class contains information on the slide collection.

        Args:
            collection_name (str): Name of the collection (i.e. Name of tumor set or patient ID)

            src_dir (str): Path to src directory containing the WSIs.

            dest_dir (str): Path to destination directory for results.

            ref_slide (str, optional): Path to reference slide. If 'ref_slide' is None it will be automatically set to
                the HE slide based on the filename of input files. Defaults to None.

            path_antigen_profiles (str, optional): Path to antigen profiles. Definitions as .json or .csv are accepted.
                If no default thresholds will be applied during processing.
        """
        # Logging
        logging.config.dictConfig(log_config.LOGGING)
        self.logger = logging.getLogger(__name__)

        # Name of the tumorset
        self.collection_name = collection_name

        # Validate directories
        if not os.path.isdir(src_dir):
            raise ValueError(
                f"Source directory {src_dir} does not exist or is not accessible."
            )
        if not os.path.isdir(dest_dir):
            raise ValueError(
                f"Destination directory {dest_dir} does not exist or is not accessible."
            )

        # Directories
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.registration_dir = None
        self.data_dir = None
        self.pickle_dir = None
        self.tiles_dir = None
        self.colocalization_dir = None
        self.reconstruct_dir = None

        # List containing all slide objects
        self.slides = []

        # Slide informations
        self.collection_info = pd.DataFrame(columns=SLIDE_COLLECTION_COLUMN_NAMES)

        # Mask Variables
        self.mask = None
        self.mask_coordinates = []

        # Reference Slide
        self.reference_slide = ref_slide

        # Quantification Variables
        self.quantification_results = pd.DataFrame(
            columns=QUANTIFICATION_RESULTS_COLUMN_NAMES
        )

        # Antigen Expression Variables
        self.dual_antigen_expressions = pd.DataFrame(
            columns=DUAL_ANTIGEN_EXPRESSIONS_COLUMN_NAMES
        )
        self.triplet_antigen_results = pd.DataFrame(
            columns=TRIPLET_ANTIGEN_EXPRESSIONS_COLUMN_NAMES
        )

        self.status = {
            "initialized": False,
            "registered": False,
            "segmented": False,
            "quantified": False,
            "dual_antigen_expression": False,
            "triplet_antigen_expression": False,
        }

        # Set destination directories
        self.set_dst_dir()

        # Initialize the slide collection
        self.init_slide_collection()

        # Load previous results if exist
        self.load_previous_results()

        if path_antigen_profiles is not None:
            self.add_antigen_profiles(path_antigen_profiles)

        # Log initialization details
        self.logger.debug(
            f"Initialized SlideCollection with collection_name: {self.collection_name}, "
            f"src_dir: {self.src_dir}, dest_dir: {self.dest_dir}, "
            f"ref_slide: {self.reference_slide}, "
            f"backend: {get_backend_info()}"
        )

    def set_dst_dir(self):
        """Assign and initiate needed directories if they do not exist yet."""
        # Data dir
        self.data_dir = os.path.join(self.dest_dir, RESULT_DATA_DIR)
        os.makedirs(self.data_dir, exist_ok=True)
        # Pickle dir
        self.pickle_dir = os.path.join(self.data_dir, PICKLE_DIR)
        os.makedirs(self.pickle_dir, exist_ok=True)

        self.logger.debug("Data and Pickle directories created")

    def init_slide_collection(self):
        """
        Initializes the slide collection by iterating over the files in the source directory. Only files with the
        extensions '.tiff' or '.tif' are considered. For each valid file, a 'Slide' object is created. Each Slide's
        information is added to the 'collection_info_df'.

        Returns:
            None
        """
        self.logger.info(f"Initializing SlideCollection: {self.collection_name}")
        init_start_time = time()
        for file in os.listdir(self.src_dir):
            if os.path.isfile(os.path.join(self.src_dir, file)):
                file_ext = os.path.splitext(file)[-1].lower()
                if not file.startswith(".") and file_ext in SUPPORTED_IMAGE_FORMATS:
                    filename = cutils.get_name(file)
                    mask = False
                    ref = False
                    # Look for mask and reference slide. If no reference selected HE slide will be selected
                    if re.search("_mask", filename):
                        mask = True
                    elif re.search("HE", filename) or filename == self.reference_slide:
                        ref = True

                    slide = Slide(
                        filename,
                        os.path.join(self.src_dir, file),
                        is_mask=mask,
                        is_reference=ref,
                    )
                    self.slides.append(slide)
                    self.collection_info.loc[len(self.collection_info)] = (
                        slide.properties.values()
                    )
                    if ref and not self.reference_slide:
                        self.reference_slide = slide
                    elif mask:
                        self.mask = slide

        self.collection_info.to_csv(
            os.path.join(self.data_dir, "collection_info.csv"),
            sep=",",
            index=False,
            header=True,
            encoding="utf-8",
        )
        self.status["initialized"] = True

        init_end_time = time()
        self.logger.debug(
            f"Slide collection initialized in {round((init_end_time - init_start_time), 2)} seconds"
        )

    def load_previous_results(self, path=None):
        """Loads results from previous processing if they exist.

        Tries to load results from previous processing. If no path is passed, the collections `pickle_dir` is used.
        Slide objects are based on `OpenSlid` which are C-type objects and cannot be stored as pickle. Therefore, each
        Slide is re-initialized in the init_slide_collection function. The function will try to load the following
        files:

            - mask_coordinates.pickle: Load mask coordinates from previous mask processing.

            - quantification_results.pickle: Load quantification results from previous processing.

            - dual_antigen_expressions.pickle: Load dual antigen overlap results from previous processing.

            - triplet_antigen_expressions.pickle: Load triplet antigen overlap results from previous processing.

        Args:
            path (str, optional): Path to directory containing pickle files. Defaults to `pickle_dir` of the slide
                collection.
        """
        prev_res_start_time = time()
        self.logger.debug("Searching for previous results")

        reg_dir = os.path.join(self.dest_dir, REGISTRATION_DIR)
        if os.path.isdir(reg_dir) and os.listdir(reg_dir):
            self.registration_dir = reg_dir
            try:
                self.logger.info("Searching for previous registration results")
                # update Slide objects to point to registered images
                self._update_slide_paths_after_registration()
                self.status["registered"] = True
                self.logger.info(
                    f"Successfully loaded registered slides into {self.collection_name}."
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to update slide paths from existing registration directory: {e}"
                )

            self.logger.info("Searching for previous segmentation results")
            self.add_mask_to_collection(reg_dir)

        self.logger.info("Searching for previous quantification results")
        if self.quantification_results.__len__() == 0:
            if path is None:
                path = self.pickle_dir
            path_mask_coord = os.path.join(path, "mask_coordinates.pickle")
            path_quant_res = os.path.join(path, "quantification_results.pickle")
            path_dual_overlap_res = os.path.join(
                path, "dual_antigen_expressions.pickle"
            )
            path_triplet_overlap_res = os.path.join(
                path, "triplet_antigen_expressions.pickle"
            )

            # load mask coordinates
            if os.path.exists(path_mask_coord):
                self.mask_coordinates = pickle.load(open(path_mask_coord, "rb"))
                self.status["segmented"] = True
                self.logger.info(
                    f"Successfully loaded mask coordinates for {self.collection_name}"
                )
            else:
                self.logger.debug(
                    f"No mask coordinates found for {self.collection_name}"
                )

            # load quantification results
            if os.path.exists(path_quant_res):
                self.quantification_results = pickle.load(open(path_quant_res, "rb"))
                self.status["quantified"] = True
                self.logger.info(
                    f"Sucessfully loaded quantification results for {self.collection_name}"
                )
            else:
                self.logger.debug(
                    f"No previous quantification results found for {self.collection_name}"
                )

            # load dual overlap results
            self.logger.info("Searching for previous dual antigen expression results")
            if os.path.exists(path_dual_overlap_res):
                self.dual_antigen_expressions = pickle.load(
                    open(path_dual_overlap_res, "rb")
                )
                self.status["dual_antigen_expression"] = True
                self.logger.info(
                    f"Successfully loaded dual_antigen_expression results for {self.collection_name}"
                )
            else:
                self.logger.debug(
                    f"No previous dual_antigen_expression results found for {self.collection_name}"
                )

            # load triplet overlap results
            self.logger.info(
                "Searching for previous triplet antigen expression results"
            )
            if os.path.exists(path_triplet_overlap_res):
                self.triplet_antigen_results = pickle.load(
                    open(path_triplet_overlap_res, "rb")
                )
                self.status["triplet_antigen_expression"] = True
                self.logger.info(
                    f"Successfully loaded triplet_antigen_expression results for {self.collection_name}"
                )
            else:
                self.logger.debug(
                    f"No previous triplet_antigen_expression results found for {self.collection_name}"
                )

            # Load processing info for each slide
            for slide in self.slides:
                path_slide = os.path.join(path, f"{slide.name}_processing_info.pickle")
                if os.path.exists(path_slide):
                    slide.detailed_quantification_results = pickle.load(
                        open(path_slide, "rb")
                    )
                    self.logger.debug(
                        f"Successfully loaded processing info for slide {slide.name}"
                    )

                    # If quantification results for loaded slide exist, load them into the slide object
                    if not self.quantification_results[
                        self.quantification_results["Name"] == slide.name
                    ].empty:
                        slide.quantification_summary = self.quantification_results.loc[
                            self.quantification_results["Name"] == slide.name
                        ]
                        self.logger.debug(
                            f"Successfully loaded detailed quantification results for slide {slide.name}"
                        )
                    else:
                        self.logger.debug(
                            f"No quantification results found for slide {slide.name}"
                        )
                else:
                    self.logger.debug(
                        f"No previous processing info found for slide {slide.name}"
                    )
                    pass
        prev_res_end_time = time()
        self.logger.info(
            f"Finished loading previous results for {self.collection_name} in \
                {round((prev_res_end_time - prev_res_start_time), 2)} seconds"
        )

    def add_mask_to_collection(self, dir):
        """Adds a mask slide to the slide collection.

        Args:
            dir (str): Path to directory the mask is in.
        """
        if not dir or not os.path.isdir(dir):
            self.logger.debug(
                f"Registration directory {dir} does not exist yet. Please run registration first."
            )
        try:
            for fname in os.listdir(dir):
                if re.search(r"_mask", fname, re.IGNORECASE):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in SUPPORTED_IMAGE_FORMATS:
                        continue
                    mask_name = cutils.get_name(fname)
                    mask_path = os.path.join(dir, fname)
                    existing = [s for s in self.slides if s.name == mask_name]
                    if existing:
                        existing[0].is_mask = True
                        try:
                            existing[0].update_slide(mask_path)
                        except Exception:
                            self.logger.debug(
                                f"Could not update existing mask slide path for {mask_name}"
                            )
                        self.mask = existing[0]
                    else:
                        try:
                            new_mask = Slide(mask_name, mask_path, is_mask=True)
                            self.slides.append(new_mask)
                            try:
                                self.collection_info.loc[len(self.collection_info)] = (
                                    new_mask.properties.values()
                                )
                            except Exception:
                                self.logger.debug(
                                    "Could not append new mask slide to collection_info"
                                )
                            self.mask = new_mask
                        except Exception as e:
                            self.logger.warning(
                                f"Found mask Slide but failed to load it: {mask_path}: {e}"
                            )
                    self.logger.info(f"Loaded mask slide from directory: {mask_path}")
                    self.status["segmented"] = True
        except Exception as e:
            self.logger.debug(f"Error while scanning {dir} for mask: {e}")

    def extract_mask_tile_coordinates(self, save_img=False):
        """Extracts mask coordinates from the mask slide.

        Generates a list containing of tiles coordinates that are part of the mask. This allows to only process tiles
        that are part of the mask and thus contain tumor tissue. Tiles with less than 10% tumor tissue are dropped due
        to save runtime. Previous mask coordinates will be overwritten and the results will be stored as pickle in
        `pickle_dir`.

        Args:
            save_img (bool): Boolean to determine if mask tiles shall be saved as image. Necessary if mask shall be
                reconstructed later on. Note: Storing tiles will require addition storage. Defaults to False.
        """
        # Create tiles directory if it does not exist
        if save_img:
            self.tiles_dir = os.path.join(self.dest_dir, TILES_DIR)
            os.makedirs(self.tiles_dir, exist_ok=True)

        # If no mask slide is provided, mask coordinates will contain all tiles of the slide.
        if self.mask is None:
            mask_start_time = time()
            slide_tiles = self.reference_slide.tiles
            self.mask_coordinates.clear()
            cols, rows = slide_tiles.level_tiles[slide_tiles.level_count - 1]
            for col in tqdm(range(cols), desc="Extracting mask tiles"):
                for row in range(rows):
                    self.mask_coordinates.append((col, row))
        else:
            mask_start_time = time()
            self.logger.debug("Extracting Mask Tile Coordinates")
            mask_tiles = self.mask.tiles
            self.mask_coordinates.clear()
            cols, rows = mask_tiles.level_tiles[mask_tiles.level_count - 1]
            for col in tqdm(range(cols), desc="Extracting mask tiles"):
                for row in range(rows):
                    temp = mask_tiles.get_tile(mask_tiles.level_count - 1, (col, row))
                    if temp.mode != "RGB":
                        temp = temp.convert("RGB")
                    temp = xp.array(temp)
                    mean = xp.mean(temp)

                    # If tile is mostly white, drop tile coordinate; cutoff: 230 (10%)
                    if mean < 230:
                        self.mask_coordinates.append((col, row))
                        if save_img:
                            tile_name = str(col) + "_" + str(row)
                            # Convert to NumPy array if necessary
                            if hasattr(temp, "get"):
                                temp = temp.get()
                            elif hasattr(temp, "asnumpy"):
                                temp = temp.asnumpy()
                            # Convert to PIL Image
                            img = Image.fromarray(temp)

                            dir = os.path.join(self.tiles_dir, "mask")
                            os.makedirs(dir, exist_ok=True)
                            out_path = os.path.join(dir, tile_name + ".tif")
                            img.save(out_path)
                    else:
                        pass
        mask_end_time = time()
        self.logger.debug(
            f"Mask coordinates generated in {round((mask_end_time - mask_start_time) / 60, 2)} minutes"
        )

        # Save mask coordinates as pickle
        out = os.path.join(self.pickle_dir, "mask_coordinates.pickle")
        with open(out, "wb") as file:
            pickle.dump(self.mask_coordinates, file, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(self.mask_coordinates, open(out, "wb"))
        self.logger.debug(f"Successfully saved mask coordinates to {out}")
        self.logger.info("Finished Mask Tile Extraction")

    def register_slides(
        self,
        reference_slide=None,
        microregistration=True,
        max_non_rigid_registration_dim_px=2000,
        crop=None,
        high_res_alignement=False,
        high_res_fraction=None,
    ):
        """Registers all WSIs in the collection using Valis.

        Registers all WSIs in the collection and aligns them. Registration can be performed towards a selected
        referenceWSI or automatically towards a WSI chosen by VALIS. Registration includes rigid and non-rigid
        registration steps. Optional `microregistration` can be applied to further improve registration quality.
        The registered slides are saved in the `registration` directory. Intermediate results are stored in the
        `intermediate_registration_results` directory. Further configuration options such as
        `max_non_rigid_registration_dim_px` and cropping methods are available.
        Lastly, an additional, customizable high-resolution alignment can be performed if specified, which allows
        tailoring the alignment resolution via the `high_res_fraction` parameter.

        Args:
            reference_slide (str): Path to reference slide. If None, the first slide in the collection will be used.
                Defaults to None.

            microregistration (bool): Boolean determining if microregistration shall be used. Defaults to True.

            max_non_rigid_registration_dim_px (int): Maximum size of non-rigid registration dimension in pixels.
                Defaults to 2000.

            crop (str): Crop method to be used. Defaults to None which results cropping to reference slide if one is
                provided. If no reference slide is provided, the default crop method is 'overlap'.

            high_res_alignment (bool): Boolean determining if customizable high-resolution alignment shall be
                performed. Defaults to False.

            high_res_fraction (float): Fraction of the image to be used for high resolution alignment. Defaults to None.
        """
        registration_begin_time = time()

        self.registration_dir = os.path.join(self.dest_dir, REGISTRATION_DIR)
        os.makedirs(self.registration_dir, exist_ok=True)

        self.intermediate_registration_dir = os.path.join(
            self.data_dir, "intermediate_registration_results"
        )
        os.makedirs(self.intermediate_registration_dir, exist_ok=True)

        if reference_slide:
            # Accept either a Slide object or a path string for reference_slide
            if isinstance(reference_slide, Slide):
                ref_path = reference_slide.orig_path
            else:
                ref_path = reference_slide
            self.logger.info(f"Registering slides with reference slide {ref_path}")
            register_slides_with_reference(
                slide_src_dir=self.src_dir,
                results_dst_dir=self.intermediate_registration_dir,
                registered_slides_dst=self.registration_dir,
                referenceSlide=ref_path,
                microregistration=microregistration,
                max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px,
                crop=crop,
            )
        elif high_res_alignement:
            self.logger.info(
                f"Performing high resolution alignment with {high_res_fraction} fraction"
            )
            register_slides_high_resolution(
                slide_src_dir=self.src_dir,
                results_dst_dir=self.registration_dir,
                registered_slides_dst=self.registration_dir,
                micro_reg_fraction=high_res_fraction,
            )
        else:
            self.logger.info("Registering slides without reference slide")
            register_slides(
                slide_src_dir=self.src_dir,
                results_dst_dir=self.intermediate_registration_dir,
                registered_slides_dst=self.registration_dir,
                microregistration=microregistration,
                max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px,
                crop=crop,
            )

        registration_end_time = time()
        self.status["registered"] = True
        self.logger.info(
            f"Finished image registration of {self.collection_name} in \
                {round((registration_end_time - registration_begin_time) / 60, 2)} minutes."
        )
        self.logger.info(f"Updating slide paths for {self.collection_name}")
        self._update_slide_paths_after_registration()

    def _update_slide_paths_after_registration(self):
        """Update Slide objects to use registered slide files saved in self.registration_dir.
        Uses original basename to look for the registered file (registration preserves filenames).
        """
        if not self.registration_dir or not os.path.isdir(self.registration_dir):
            self.logger.info(
                "Registration directory not set or missing; skipping _update_slide_paths_after_registration."
            )
            return
        # First, scan all files in registration_dir recursively and build a mapping from stem -> path
        reg_files = {}
        for root, _, files in os.walk(self.registration_dir):
            for f in files:
                stem = cutils.get_name(f)
                path = os.path.join(root, f)
                reg_files[stem] = path

        for slide in self.slides:
            base_stem = cutils.get_name(slide.orig_path)
            if base_stem in reg_files:
                reg_path = reg_files[base_stem]
                try:
                    slide.update_slide(reg_path)
                except Exception as e:
                    self.logger.error(
                        f"Could not update slide {slide.name} to {reg_path}: {e}"
                    )
            else:
                self.logger.debug(
                    f"No registered file found for {slide.name} (stem={base_stem})"
                )

    def tumor_segmentation(
        self,
        model_path,
        reference_slide=None,
        tile_size=[1024, 1024],
        output_path=None,
        normalization=False,
        inversion=False,
        plot_results=False,
        batch_size=None,
        num_workers=None,
        gpu_ids=None,
    ):
        """Performs tumor segmentation on the HE WSI in the SlideCollection.

        Performs tumor segmentation on either a specified reference WSI or the reference WSI selected by the
        SlideCollection. The segmentation model needs to be passed. By default, segmentation results are saved into the
        `registration_dir` of the SlideCollection, an optional ouput path can be provided. Model-specific parameters
        such as `normalization`, or `inversion` can also be passed. Lastly, segmentation_results can be plotted onto
        the tissue if `plot_results` is set to True.

        Args:
            model_path (str): Path to segmentation model.

            reference_slide (str, optional): Path to reference slide. If None, the reference slide of the
                SlideCollection is used. Defaults to None.

            tile_size (list, optional): Tile size to be used for segmentation. Defaults to [1024, 1024].

            output_path (str, optional): Output path for segmentation results. If None, the `registration_dir` of the
                SlideCollection is used. Defaults to None.

            normalization (bool, optional): Boolean determining if stain normalization shall be applied. Defaults to
                False.

            inversion (bool, optional): Boolean determining if color inversion shall be applied. Defaults to False.

            plot_results (bool, optional): Boolean determining if segmentation results shall be plotted onto the tissue.
                Defaults to False.

            batch_size (int, optional): Batch size for segmentation. Defaults to None.

            num_workers (int, optional): Number of workers for segmentation. Defaults to None.

            gpu_ids (list, optional): List of GPU IDs for segmentation. Defaults to None.

        Returns:
            None
        """

        if reference_slide is None:
            reference_slide = self.reference_slide.registered_path

        if output_path is None:
            output_path = self.registration_dir

        run_tumor_segmentation(
            reference_slide,
            model_path,
            tile_size,
            output_path,
            normalization,
            inversion,
            plot_results,
            batch_size,
            num_workers,
            gpu_ids,
        )
        self.add_mask_to_collection(output_path)
        self.status["segmented"] = True

    def add_antigen_profiles(self, profile_path):
        """Adds antigen profiles to slides in the collection based on matching names.

        Args:
            profile_path (str): Path to antigen profile file. Supported formats are CSV and JSON.
        """
        if profile_path.endswith(".csv"):
            profiles_df = pd.read_csv(profile_path)
        elif profile_path.endswith(".json"):
            profiles_df = pd.read_json(profile_path)
        else:
            raise ValueError(
                "Unsupported file format for antigen profile. Must be CSV or JSON"
            )

        if "Name" not in profiles_df.columns:
            raise ValueError("Antigen profile must contain a Name")

        for slide in self.slides:
            if slide.is_mask or slide.is_reference:
                continue
            for _, profile in profiles_df.iterrows():
                antigen_name = str(profile["Name"]).lower()
                if antigen_name in slide.name.lower():
                    # print(profile.to_dict())
                    slide.antigen_profile.update(profile.to_dict())
                    self.logger.debug(
                        f"Antigen profile'{antigen_name}' added to slide {slide.name}"
                    )
                    break
                else:
                    self.logger.debug(
                        f"No antigen profile found matching slide {slide.name}"
                    )

    def quantify_all_slides(self, save_imgs=False, masking_mode="tile-level"):
        """Quantifies all registered slides sequentially and stores results.

        Quantifies all slides that were instantiated sequentially with the exception of the reference_slide and the
        mask_slide. Results are stored as .CSV into the `data_dir`. All previous quantification results in the
        `quant_res_df` will be reset and the previous .CSV file overwritten.

        Args:
            save_imgs (bool): Boolean determining if tiles shall be saved as image during processing. This is necessary
                if slides shall be reconstructed after processing. Note: storing tiles may require substantial
                additional storage. Defaults to False.

            masking_mode (str): Defines how the tumor mask is applied to tiles during quantification.

                - `tile-level` (default): Applies the mask coarsly - tiles overlapping the mask are fully included.
                  Recommended when registration quality is lower (e.g. high median rTRE).

                - `pixel-level`: Applies the mask precisely at pixel level - only masked pixels are included.
                  Offers finer quantification, but is more sensitive to registration errors.
        """
        if masking_mode not in ["tile-level", "pixel-level"]:
            raise ValueError(
                f"masking_mode must either be 'tile-level' or 'pixel-level'. {masking_mode} is not supported."
            )
        start_quant_time = time()
        self.logger.info(
            f"Starting quantification of all slides, save_imgs: {save_imgs}, masking_mode: {masking_mode}"
        )
        if self.quantification_results.__len__() != 0:
            self.quantification_results = self.quantification_results.iloc[0:0]
        # Counter variable for progress tracking
        c = 1
        for slide in self.slides:
            if not slide.is_mask and not slide.is_reference:
                self.logger.info(
                    f"Analyzing Slide: {slide.name}({c}/{len(self.slides) - 2})"
                )
                self.quantify_single_slide(slide.name, save_imgs, masking_mode)
                c += 1
        end_quant_time = time()
        self.logger.info(f"Finished quantification for {self.collection_name} in \
                {round((end_quant_time - start_quant_time) / 60, 2)} minutes.")
        self.status["quantified"] = True

    def quantify_single_slide(
        self, slide_name, save_img=False, masking_mode="tile-level"
    ):
        """Quantifies a single slide and appends the results to `quant_res_df`.

        This function quantifies staining intensities for all tiles of the given slide using multiprocessing. The slide
        matching the passed slide_name is retrieved from the collection_list and quantified using the quantify_slide
        function of the Slide class. Results are appended to `quant_res_df`, which is then stored as .CSV in self.
        `data_dir` and as .PICKLE in `pickle_dir`. Existing .CSV/.PICKLE files are overwritten.
        For more information on quantification checkout Slide.quantify_slide() function in the slide.py.

        Args:
            slide_name (str): Name of the slide to be processed.

            save_img (bool): Boolean determining if tiles shall be saved during processing. Necessary if slide shall be
                reconstructed later on. However, storing images will require addition storage. Defaults to False.

            masking_mode (str): Defines how the tumor mask is applied to tiles during quantification.

                - `tile-level` (default): Applies the mask coarsly - tiles overlapping the mask are fully included.
                   Recommended when registration quality is lower (e.g. high median rTRE).

                - `pixel-level`: Applies the mask precisely at pixel level - only masked pixels are included.
                   Offers finer quantification, but is more sensitive to registration errors.
        """
        if masking_mode not in ["tile-level", "pixel-level"]:
            raise ValueError(
                f"masking_mode must either be 'tile-level' or 'pixel-level'. {masking_mode} is not supported."
            )
        if self.status["registered"] is False:
            raise RuntimeError(
                "Slides must be registered before quantification. Please call register_slides() before quantification."
            )
        if self.status["segmented"] is False:
            raise RuntimeError(
                "Tumor segmentation must be performed before quantification.\
                      Please call tumor_segmentation() before quantification."
            )

        if not self.mask_coordinates:
            self.logger.info("Extracting mask coordinates before quantification")
            self.extract_mask_tile_coordinates()

        slide = [slide for slide in self.slides if slide.name == slide_name][0]

        # Resolve mask path only for pixel-level masking
        mask_path = None
        if masking_mode == "pixel-level":
            if self.mask is None:
                raise RuntimeError(
                    "Pixel-level masking requires a tumor mask. Please run tumor_segmentation() first."
                )
            mask_path = self.mask.orig_path or self.mask.registered_path
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            self.logger.debug(f"Using mask path: {mask_path}")

        # Directory for saved DAB tiles (only if tiles should be saved)
        dab_tile_dir = (
            os.path.join(self.tiles_dir, slide_name, DAB_TILE_DIR) if save_img else None
        )

        slide.quantify_slide(
            self.mask_coordinates,
            self.pickle_dir,
            save_img=save_img,
            img_dir=dab_tile_dir,
            mask_path=mask_path,
        )

        # slide_summary_series = pd.Series(slide.quantification_summary)
        slide_name_summary = slide.quantification_summary["Name"]

        # Check if a row with the same 'Name' exists
        existing_row_index = self.quantification_results[
            self.quantification_results["Name"] == slide_name_summary
        ].index

        if not existing_row_index.empty:
            # Overwrite the existing row
            self.quantification_results.loc[existing_row_index[0]] = pd.Series(
                slide.quantification_summary
            )
        else:
            # Append the new row
            self.quantification_results = pd.concat(
                [
                    self.quantification_results,
                    pd.DataFrame([slide.quantification_summary]),
                ],
                ignore_index=True,
            )

        # Sort the DataFrame by the 'Name' column
        self.quantification_results = self.quantification_results.sort_values(
            by="Coverage (%)", ascending=False
        ).reset_index(drop=True)

        self.save_quantification_results(masking_mode=masking_mode)

    def save_quantification_results(self, masking_mode):
        """
        Stores `quant_res_df` as .CSV for and .PICKLE in `data_dir` and `pickle_dir`, respectively.
        """
        if self.quantification_results.__len__() != 0:
            save_start_time = time()
            filename = f"{self.data_dir}/{masking_mode}_quantification_results.csv"
            self.quantification_results.to_csv(
                filename,
                sep=",",
                index=False,
                encoding="utf-8",
            )
            out = os.path.join(self.pickle_dir, "quantification_results.pickle")
            with open(out, "wb") as file:
                pickle.dump(
                    self.quantification_results, file, protocol=pickle.HIGHEST_PROTOCOL
                )
            save_end_time = time()
            self.logger.debug(f"Successfully saved quantification results to {out} in \
                    {round((save_end_time - save_start_time), 2)} seconds")
        else:
            self.logger.warning(
                "No quantification results were found. Please call quantify_all_slides() to quantify all slides \
                    in this slide collection or call quantify_single_slide() to quantify a single slide."
            )

    def generate_antigen_pair_combinations(self, masking_mode="tile-level", save_img=False):
        """Creates all possible antigen pairs and analyzes antigen co-expression for all pairs. Results are stored
        in `dual_antigen_expressions`.

        Args:
            masking_mode (str): Determines the mode for mask application that was used for quantification.

                - `tile-level` (default): Applies the mask coarsly - tiles overlapping the mask are fully included.
                   Recommended when registration quality is lower (e.g. high median rTRE).

                - `pixel-level`: Applies the mask precisely at pixel level - only masked pixels are included.
                   Offers finer co-expression evaluation, but is more sensitive to registration errors.

                - `save_img` (bool): Boolean determining if tiles shall be saved during processing. Necessary if slide
                    shall be reconstructed later on. However, will require additional storage. Defaults to False.
        """
        dual_expression_time_start = time()
        self.dual_antigen_expressions = pd.DataFrame(
            columns=DUAL_ANTIGEN_EXPRESSIONS_COLUMN_NAMES
        )
        # Filter out mask and reference slides
        filtered_slides = [
            slide
            for slide in self.slides
            if not slide.is_mask and not slide.is_reference
        ]

        # Generate all possible pairs of the filtered slides
        slide_combinations = list(combinations(filtered_slides, 2))

        # Pass each combination to the compute_dual_antigen_combination method
        for combo in slide_combinations:
            self.evaluate_antigen_pair(combo[0], combo[1], masking_mode=masking_mode, save_img=save_img)
        dual_expression_time_end = time()
        self.logger.info(
            f"Finished dual antigen expression analysis in \
                {round((dual_expression_time_end - dual_expression_time_start) / 60, 2)} minutes."
        )
        self.status["dual_antigen_expression"] = True

    def generate_antigen_triplet_combinations(self, masking_mode="tile-level", save_img=False):
        """Creates all possible antigen triplets and analyzes antigen co-expression for all triplets. Results are stored
        in `triplet_antigen_expressions`.

        Args:
            masking_mode (str): Determines the mode for mask application that was used for quantification.

                - `tile-level` (default): Applies the mask coarsly - tiles overlapping the mask are fully included.
                   Recommended when registration quality is lower (e.g. high median rTRE).

                - `pixel-level`: Applies the mask precisely at pixel level - only masked pixels are included.
                   Offers finer co-expression evaluation, but is more sensitive to registration errors.

                - `save_img` (bool): Boolean determining if tiles shall be saved during processing. Necessary if slide
                    shall be reconstructed later on. However, will require additional storage. Defaults to False.
        """
        triplet_expression_time_start = time()
        self.triplet_antigen_results = pd.DataFrame(
            columns=TRIPLET_ANTIGEN_EXPRESSIONS_COLUMN_NAMES
        )
        # Filter out mask and reference slides
        filtered_slides = [
            slide
            for slide in self.slides
            if not slide.is_mask and not slide.is_reference
        ]

        # Generate all possible triplets of the filtered slides
        slide_combinations = list(combinations(filtered_slides, 3))

        # Pass each combination to the compute_triplet_antigen_combinations method
        for combo in slide_combinations:
            self.evaluate_antigen_triplet(
                combo[0], combo[1], combo[2], masking_mode=masking_mode, save_img=save_img
            )
        triplet_expression_time_end = time()
        self.logger.info(
            f"Finished triplet antigen expression analysis in \
                {round((triplet_expression_time_end - triplet_expression_time_start) / 60, 2)} minutes."
        )
        self.status["triplet_antigen_expression"] = True

    def evaluate_antigen_pair(
        self, slide1, slide2, save_img=False, masking_mode="tile-level"
    ):
        """Analyzes the antigen co-expression for a pair of two slides.

        Analyzes antigen co-expressions for each tile for the given pair of slides using multiprocesing based on the
        antigen-specific thresholds of each slide. Results from each of the tiles are summarized, stored in
        `dual_antigen_expressions` and saved as CSV in `data_dir` as well as PICKLE in `pickle_dir`. For more detailed
        explanation of the co-expression results see `SlideCollection.triplet_antigen_expressions`.

        Args:
            slide1 (Slide): Slide Object for slide 1.

            slide2 (Slide): Slide Object for slide 2.

            save_img (bool):  Boolean determining if tiles shall be saved during processing. Necessary if slide shall
                be reconstructed later on. However, storing images will require additional storage. Defaults to False.

            masking_mode (str): Determines the mode for mask application that was used for quantification.

                - `tile-level` (default): Applies the mask coarsly - tiles overlapping the mask are fully included.
                   Recommended when registration quality is lower (e.g. high median rTRE).

                - `pixel-level`: Applies the mask precisely at pixel level - only masked pixels are included.
                   Offers finer co-expression evaluation, but is more sensitive to registration errors.
        """
        # Create directory for pair of slides
        if save_img:
            # Create Colocalization directory if it does not exist
            self.colocalization_dir = os.path.join(self.dest_dir, COLOCALIZATION)
            os.makedirs(self.colocalization_dir, exist_ok=True)

            # Create sub-directory for slide pair
            dir = os.path.join(
                self.colocalization_dir, (slide1.name + "_and_" + slide2.name)
            )
            os.makedirs(dir, exist_ok=True)
        else:
            dir = None

        # Create iterable for multiprocessing
        iterable = []
        for i in slide1.detailed_quantification_results:
            if (
                slide1.detailed_quantification_results[i]["Tilename"]
                == slide2.detailed_quantification_results[i]["Tilename"]
            ):
                _dict1 = slide1.detailed_quantification_results[i]
                _dict2 = slide2.detailed_quantification_results[i]
                iterable.append(
                    (
                        _dict1,
                        _dict2,
                        [slide1.antigen_profile, slide2.antigen_profile],
                        dir,
                        save_img,
                        masking_mode,
                    )
                )
        start_time = time()
        self.logger.debug(
            f"Starting antigen analysis for: {slide1.name} & {slide2.name}"
        )

        # Init dict for results of each tile
        comparison_dict = {}
        max_workers = os.cpu_count() - 1
        mp_context = mp.get_context("spawn")  # Use 'spawn' context for multiprocessing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as exe:
            results = tqdm(
                exe.map(
                    evaluate_antigen_pair_tile,
                    iterable,
                ),
                total=len(iterable),
                desc="Calculating Coverage of Slide "
                + slide1.name
                + " & "
                + slide2.name,
            )
            for idx, result in enumerate(results):
                comparison_dict[idx] = result
        end_time = time()
        if end_time - start_time >= 60:
            self.logger.debug(
                f"Finished antigen analysis for: {slide1.name} & {slide2.name} in \
                    {round((end_time - start_time) / 60, 2)} minutes."
            )
        else:
            self.logger.debug(
                f"Finished antigen analysis for: {slide1.name} & {slide2.name} in \
                    {round((end_time - start_time), 2)} seconds."
            )
        self.summarize_antigen_combinations(
            comparison_dict,
            [slide1.name, slide2.name],
            antigen_profiles=[slide1.antigen_profile, slide2.antigen_profile],
        )
        self.save_antigen_combinations(result_type="dual", masking_mode=masking_mode)

    def evaluate_antigen_triplet(
        self, slide1, slide2, slide3, save_img=False, masking_mode="tile-level"
    ):
        """Analyzes the antigen co-expression for a triplet of three slides.

        Analyzes antigen co-expressions for each of tiles of the given triplet of slides using Multiprocessing based on
        the antigen-specific thresholds of each slide. Results from each of the tiles are summarized, stored in
        `triplet_antigen_expressions` and saved as CSV in `data_dir` as well as PICKLE in `pickle_dir`. For more
        detailed explanation of the co-expression results see `SlideCollection.triplet_antigen_expressions`.

        Args:
            slide1 (Slide): Slide Object for slide 1.

            slide2 (Slide): Slide Object for slide 2.

            slide3 (Slide): Slide Object for slide 3.

            save_img (bool):  Boolean determining if tiles shall be saved during processing. Necessary if slide shall
                be reconstructed later on. However, storing images will require additional storage. Defaults to False.

            masking_mode (str): Determines the mode for mask application that was used for quantification.

                - `tile-level` (default): Applies the mask coarsly - tiles overlapping the mask are fully included.
                   Recommended when registration quality is lower (e.g. high median rTRE).

                - `pixel-level`: Applies the mask precisely at pixel level - only masked pixels are included.
                   Offers finer co-expression evaluation, but is more sensitive to registration errors.
        """

        # Create directory for triplet of slides
        if save_img:
            # Create Colocalization directory if it does not exist
            self.colocalization_dir = os.path.join(self.dest_dir, COLOCALIZATION)
            os.makedirs(self.colocalization_dir, exist_ok=True)

            # Create sub-directory for slide triplet
            dirname = os.path.join(
                self.colocalization_dir,
                (slide1.name + "_and_" + slide2.name + "_and_" + slide3.name),
            )
            os.makedirs(dirname, exist_ok=True)
        else:
            dirname = None

        # Create iterable for multiprocessing
        iterable = []
        for i in slide1.detailed_quantification_results:
            if (
                slide1.detailed_quantification_results[i]["Tilename"]
                == slide2.detailed_quantification_results[i]["Tilename"]
                == slide3.detailed_quantification_results[i]["Tilename"]
            ):
                _dict1 = slide1.detailed_quantification_results[i]
                _dict2 = slide2.detailed_quantification_results[i]
                _dict3 = slide3.detailed_quantification_results[i]
                iterable.append(
                    (
                        _dict1,
                        _dict2,
                        _dict3,
                        [
                            slide1.antigen_profile,
                            slide2.antigen_profile,
                            slide3.antigen_profile,
                        ],
                        dirname,
                        save_img,
                        masking_mode,
                    )
                )

        # Init dict for results of each tile
        start_time = time()
        self.logger.debug(
            f"Starting antigen analysis for: {slide1.name} & {slide2.name} & {slide3.name}"
        )
        comparison_dict = {}
        max_workers = os.cpu_count() - 1
        # Process tiles using multiprocessing
        mp_context = mp.get_context("spawn")  # Use 'spawn' context for multiprocessing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as exe:
            results = tqdm(
                exe.map(evaluate_antigen_triplet_tile, iterable),
                total=len(iterable),
                desc="Calculating Coverage of Slides "
                + slide1.name
                + " & "
                + slide2.name
                + " & "
                + slide3.name,
            )
            for idx, result in enumerate(results):
                comparison_dict[idx] = result
        end_time = time()
        if end_time - start_time >= 60:
            self.logger.debug(
                f"Finished antigen analysis for: {slide1.name} & {slide2.name} & {slide3.name} in \
                    {round((end_time - start_time) / 60, 2)} minutes."
            )
        else:
            self.logger.debug(
                f"Finished antigen analysis for: {slide1.name} & {slide2.name} & {slide3.name} in \
                    {round((end_time - start_time), 2)} seconds."
            )
        self.summarize_antigen_combinations(
            comparison_dict,
            [slide1.name, slide2.name, slide3.name],
            antigen_profiles=[
                slide1.antigen_profile,
                slide2.antigen_profile,
                slide3.antigen_profile,
            ],
        )
        self.save_antigen_combinations(result_type="triplet", masking_mode=masking_mode)

    def summarize_antigen_combinations(
        self, comparison_dict, slide_names, antigen_profiles
    ):
        """Summarizes co-expression results of an antigen pair or triplet.

        Args:
            - comparison_dict (dict): Dictionary containing the co-expression results for each tile of the combination.

            - slide_names (str): Names of all slides of the combination.

            - antigen_profiles (list): List of all antigen profiles of the slides in the combination.
        """
        processed_tiles = 0
        sum_total_coverage_weighted = 0.00
        sum_total_overlap_weighted = 0.00
        sum_total_complement_weighted = 0.00
        sum_high_overlap_weighted = 0.00
        sum_high_complement_weighted = 0.00
        sum_pos_overlap_weighted = 0.00
        sum_pos_complement_weighted = 0.00
        sum_low_overlap_weighted = 0.00
        sum_low_complement_weighted = 0.00
        sum_negative_weighted = 0.00
        sum_tissue_weighted = 0.00
        sum_background_weighted = 0.00
        sum_mask_area = 0.00
        sum_non_mask = 0.00
        error1 = 0
        error2 = 0

        for i in comparison_dict:
            if comparison_dict[i]["Flag"] == 1:
                processed_tiles += 1
                tile_mask_pixels = comparison_dict[i]["Mask Area"] / 100 * 1048576
                sum_mask_area += tile_mask_pixels
                sum_total_coverage_weighted += (
                    comparison_dict[i]["Total Coverage"] / 100 * tile_mask_pixels
                )
                sum_total_overlap_weighted += (
                    comparison_dict[i]["Total Overlap"] / 100 * tile_mask_pixels
                )
                sum_total_complement_weighted += (
                    comparison_dict[i]["Total Complement"] / 100 * tile_mask_pixels
                )
                sum_high_overlap_weighted += (
                    comparison_dict[i]["High Positive Overlap"] / 100 * tile_mask_pixels
                )
                sum_high_complement_weighted += (
                    comparison_dict[i]["High Positive Complement"]
                    / 100
                    * tile_mask_pixels
                )
                sum_pos_overlap_weighted += (
                    comparison_dict[i]["Medium Positive Overlap"]
                    / 100
                    * tile_mask_pixels
                )
                sum_pos_complement_weighted += (
                    comparison_dict[i]["Medium Positive Complement"]
                    / 100
                    * tile_mask_pixels
                )
                sum_low_overlap_weighted += (
                    comparison_dict[i]["Low Positive Overlap"] / 100 * tile_mask_pixels
                )
                sum_low_complement_weighted += (
                    comparison_dict[i]["Low Positive Complement"]
                    / 100
                    * tile_mask_pixels
                )
                sum_negative_weighted += (
                    comparison_dict[i]["Negative"] / 100 * tile_mask_pixels
                )
                sum_tissue_weighted += (
                    comparison_dict[i]["Tissue"] / 100 * tile_mask_pixels
                )
                sum_background_weighted += (
                    comparison_dict[i]["Background / No Tissue"]
                    / 100
                    * tile_mask_pixels
                )
                sum_non_mask += comparison_dict[i]["Non-mask Area"]
            elif comparison_dict[i].get("Flag") == -1:
                error1 += 1
            elif comparison_dict[i].get("Flag") == -2:
                error2 += 1

        if processed_tiles > 0 and sum_mask_area > 0:
            sum_total_coverage = sum_total_coverage_weighted / sum_mask_area * 100
            sum_total_overlap = sum_total_overlap_weighted / sum_mask_area * 100
            sum_total_complement = sum_total_complement_weighted / sum_mask_area * 100
            sum_high_overlap = sum_high_overlap_weighted / sum_mask_area * 100
            sum_high_complement = sum_high_complement_weighted / sum_mask_area * 100
            sum_pos_overlap = sum_pos_overlap_weighted / sum_mask_area * 100
            sum_pos_complement = sum_pos_complement_weighted / sum_mask_area * 100
            sum_low_overlap = sum_low_overlap_weighted / sum_mask_area * 100
            sum_low_complement = sum_low_complement_weighted / sum_mask_area * 100
            sum_negative = sum_negative_weighted / sum_mask_area * 100
            sum_tissue = sum_tissue_weighted / sum_mask_area * 100
            sum_background = sum_background_weighted / sum_mask_area * 100
            sum_mask = (
                sum_mask_area / (processed_tiles * 1048576) * 100
            )  # average mask %
            sum_non_mask /= processed_tiles
        else:
            sum_total_coverage = 0
            sum_total_overlap = 0
            sum_total_complement = 0
            sum_high_overlap = 0
            sum_high_complement = 0
            sum_pos_overlap = 0
            sum_pos_complement = 0
            sum_low_overlap = 0
            sum_low_complement = 0
            sum_negative = 0
            sum_tissue = 0
            sum_background = 0
            sum_mask = 0
            sum_non_mask = 100

        total_error = error1 + error2
        total_processed_tiles = (
            (processed_tiles / len(comparison_dict)) * 100
            if len(comparison_dict) > 0
            else 0
        )

        overlap_dict = {"Slide 1": slide_names[0], "Slide 2": slide_names[1]}

        if len(slide_names) == 3:
            overlap_dict["Slide 3"] = slide_names[2]

        overlap_dict.update(
            {
                "Total Coverage (%)": round(float(sum_total_coverage), 2),
                "Total Overlap (%)": round(float(sum_total_overlap), 2),
                "Total Complement (%)": round(float(sum_total_complement), 2),
                "High Positive Overlap (%)": round(float(sum_high_overlap), 2),
                "High Positive Complement (%)": round(float(sum_high_complement), 2),
                "Medium Positive Overlap (%)": round(float(sum_pos_overlap), 2),
                "Medium Positive Complement (%)": round(float(sum_pos_complement), 2),
                "Low Positive Overlap (%)": round(float(sum_low_overlap), 2),
                "Low Positive Complement (%)": round(float(sum_low_complement), 2),
                "Negative Tissue (%)": round(float(sum_negative), 2),
                "Total Tissue (%)": round(float(sum_tissue), 2),
                "Background / No Tissue (%)": round(float(sum_background), 2),
                "Mask Area (%)": round(float(sum_mask), 2),
                "Non-mask Area (%)": round(float(sum_non_mask), 2),
                "Total Processed Tiles (%)": round(float(total_processed_tiles), 2),
                "Total Error (%)": round(
                    float((total_error / len(comparison_dict)) * 100), 2
                ),
                "Error1 (%)": round(float((error1 / len(comparison_dict)) * 100), 2),
                "Error2 (%)": round(float((error2 / len(comparison_dict)) * 100), 2),
                "Thresholds1": [
                    antigen_profiles[0]["high_positive_threshold"],
                    antigen_profiles[0]["medium_positive_threshold"],
                    antigen_profiles[0]["low_positive_threshold"],
                    235,
                ],
                "Thresholds2": [
                    antigen_profiles[1]["high_positive_threshold"],
                    antigen_profiles[1]["medium_positive_threshold"],
                    antigen_profiles[1]["low_positive_threshold"],
                    235,
                ],
            }
        )
        if len(antigen_profiles) == 3:
            overlap_dict["Thresholds3"] = [
                antigen_profiles[2]["high_positive_threshold"],
                antigen_profiles[2]["medium_positive_threshold"],
                antigen_profiles[2]["low_positive_threshold"],
                235,
            ]

        if len(slide_names) == 2:
            self.dual_antigen_expressions = pd.concat(
                [self.dual_antigen_expressions, pd.DataFrame([overlap_dict])],
                ignore_index=True,
            )
            self.dual_antigen_expressions = self.dual_antigen_expressions.sort_values(
                by="Total Coverage (%)", ascending=False
            )
        else:
            self.triplet_antigen_results = pd.concat(
                [self.triplet_antigen_results, pd.DataFrame([overlap_dict])],
                ignore_index=True,
            )
            self.triplet_antigen_results = self.triplet_antigen_results.sort_values(
                by="Total Coverage (%)", ascending=False
            )

    def save_antigen_combinations(self, result_type="dual", masking_mode="tile-level"):
        """
        Save antigen combination results as CSV and PICKLE.

        Args:
            result_type (str): Type of antigen combination results to save. Can be "dual" or "triplet".

            masking_mode (str): Mode of mask application for consistent file naming with the applied mode.
        """
        if result_type == "dual":
            summary_df = self.dual_antigen_expressions
            csv_filename = f"{masking_mode}_dual_antigen_expressions.csv"
            pickle_filename = "dual_antigen_expressions.pickle"
        elif result_type == "triplet":
            summary_df = self.triplet_antigen_results
            csv_filename = f"{masking_mode}_triplet_antigen_expressions.csv"
            pickle_filename = "triplet_antigen_expressions.pickle"
        else:
            raise ValueError("Invalid result_type. Must be 'dual' or 'triplet'.")

        # Save results as CSV
        summary_df.to_csv(
            os.path.join(self.data_dir, csv_filename),
            sep=",",
            index=False,
            header=True,
            encoding="utf-8",
        )

        # Save results as PICKLE
        out = os.path.join(self.pickle_dir, pickle_filename)
        with open(out, "wb") as f:
            pickle.dump(summary_df, f, protocol=pickle.HIGHEST_PROTOCOL)
