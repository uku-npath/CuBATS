# Standard Library
import os
from time import time

# Third Party
import numpy as np
from valis import registration

SLIDE_SRC_DIR = ""
RESULTS_DST_DIR = ""
REGISTERED_SLIDE_DEST_DIR = ""
REFERENCE_SLIDE = ""
DEFAULT_MAX_NON_RIGID_REG_SIZE = 2000
CROP_OVERLAP = "overlap"
CROP_REFERENCE = "reference"


def register_slides(
    slide_src_dir,
    results_dst_dir,
    registered_slides_dst=None,
    microregistration=False,
    max_non_rigid_registration_dim_px=DEFAULT_MAX_NON_RIGID_REG_SIZE,
    crop=None,
):
    """
    Register the slides using Valis. This function automatically registers the slides and saves the registered slides
    in the specified directory.

    Args:
        slide_src_dir (str): Path to input slides.

        results_dst_dir (str): Path to directory where results will be saved.

        registered_slides_dst (str, optional): Path to directory where registered slides will be saved. If no path is
            provided, the registered slides will be saved in the results directory under "registered_slides".

        microregistration (bool, optional): Whether to perform microregistration after rigid and non-rigid
            registration. Defaults to False.

        max_non_rigid_registration_dim_px (int, optional): Maximum size of the non-rigid registration dimension in
            pixels. Defaults to 2000.

        crop (str, optional): Crop type for the registered slides. Can be "overlap" or "reference". Defaults to
            "overlap".
    """
    # Input validation
    if not isinstance(slide_src_dir, str) or not os.path.exists(slide_src_dir):
        raise ValueError("Invalid or non-existent source directory")
    if not isinstance(results_dst_dir, str):
        raise ValueError("Invalid destination directory")
    if not isinstance(microregistration, bool):
        raise ValueError("microregistration must be a boolean")
    if not isinstance(max_non_rigid_registration_dim_px, int):
        raise ValueError("max_non_rigid_registration_dim_px must be an integer")
    if not isinstance(crop, str):
        raise ValueError("crop must be a string")
    if crop not in [CROP_OVERLAP, CROP_REFERENCE, None]:
        raise ValueError(f"crop must be one of {CROP_OVERLAP} or {CROP_REFERENCE}")

    if registered_slides_dst is None:
        registered_slides_dst = os.path.join(results_dst_dir, "registered_slides")

    if crop is None:
        crop = CROP_OVERLAP
    # Start registration
    registrar = registration.Valis(slide_src_dir, results_dst_dir)
    # Rigid and non-rigid registration
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    # Micro registration
    if microregistration:
        registrar.register_micro(
            max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px
        )
    # Save registered slides
    registrar.warp_and_save_slides(registered_slides_dst, crop=crop, tile_wh=1024)
    registration.kill_jvm()


def register_slides_with_reference(
    slide_src_dir,
    results_dst_dir,
    referenceSlide,
    registered_slides_dst=None,
    microregistration=False,
    max_non_rigid_registration_dim_px=DEFAULT_MAX_NON_RIGID_REG_SIZE,
    crop=None,
):
    """
    Register the slides with a reference slide using Valis. This function automatically registers the slides and saves
    the registered slides in the specified directory.

    Args:
        slide_src_dir (str): Path to input slides

        results_dst_dir (str): Path to directory where results will be saved

        referenceSlide (str): Path to reference slide

        registered_slides_dst (str, optional): Path to directory where registered slides will be saved. If no path is
            provided, the registered slides will be saved in the results directory under "registered_slides".

        microregistration (bool, optional): Whether to perform microregistration after rigid and non-rigid
            registration. Defaults to False.

        max_non_rigid_registration_dim_px (int, optional): Maximum size of the non-rigid registration dimension in
            pixels. Defaults to 2000.

        crop (str, optional): Crop type for the registered slides. Can be "overlap" or "reference". Defaults to
            "reference".
    """
    # Input validation
    if not isinstance(slide_src_dir, str) or not os.path.exists(slide_src_dir):
        raise ValueError("Invalid or non-existent source directory")
    if not isinstance(results_dst_dir, str):
        raise ValueError("Invalid destination directory")
    if not isinstance(referenceSlide, str) or not os.path.exists(referenceSlide):
        raise ValueError("Invalid or non-existent reference slide")
    if not isinstance(microregistration, bool):
        raise ValueError("microregistration must be a boolean")
    if not isinstance(max_non_rigid_registration_dim_px, int):
        raise ValueError("max_non_rigid_registration_dim_px must be an integer")

    if registered_slides_dst is None:
        registered_slides_dst = os.path.join(results_dst_dir, "registered_slides")

    if crop is None:
        crop = CROP_REFERENCE

    registrar = registration.Valis(
        slide_src_dir,
        results_dst_dir,
        reference_img_f=referenceSlide,
        align_to_reference=True,
    )
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()
    if microregistration:
        registrar.register_micro(
            max_non_rigid_registration_dim_px=max_non_rigid_registration_dim_px,
            align_to_reference=True,
        )
    registrar.warp_and_save_slides(registered_slides_dst, crop=crop, tile_wh=1024)
    registration.kill_jvm()


def register_slides_high_resolution(
    slide_src_dir,
    results_dst_dir,
    registered_slides_dst=None,
    micro_reg_fraction=None,
):
    """
    Performs high resolution alignment.
    """
    # Perform high resolution rigid registration using the MicroRigidRegistrar
    start = time()
    if registered_slides_dst is None:
        registered_slides_dst = os.path.join(results_dst_dir, "registered_slides")
    if micro_reg_fraction is None:
        micro_reg_fraction = 0.25

    registrar = registration.Valis(slide_src_dir, results_dst_dir)
    # , micro_rigid_registrar_cls=MicroRigidRegistrar
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration
    # on an image that is 25% full resolution.
    img_dims = np.array(
        [
            slide_obj.slide_dimensions_wh[0]
            for slide_obj in registrar.slide_dict.values()
        ]
    )
    min_max_size = np.min([np.max(d) for d in img_dims])
    img_areas = [np.multiply(*d) for d in img_dims]
    max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
    micro_reg_size = np.floor(min_max_size * micro_reg_fraction).astype(int)
    print(micro_reg_size)
    # Perform high resolution non-rigid registration using 25% full resolution
    micro_reg, micro_error = registrar.register_micro(
        max_non_rigid_registration_dim_px=micro_reg_size
    )
    registrar.warp_and_save_slides(
        registered_slides_dst, crop=CROP_OVERLAP, tile_wh=1024
    )
    registration.kill_jvm()
    end = time()
    print(f"High-resolution alignement completed in {end-start:.2f} seconds")
