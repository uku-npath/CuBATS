# Standard Library
import logging
import os
from time import time

# Third Party
import pyvips
from tqdm import tqdm

DEFAULT_TILE_SIZE = 1024
DEFAULT_BACKGROUND = 192


def get_deepzoom_grid(slide):
    """Return ((cols, rows), (width, height)) of the top DeepZoom level of a slide.

    Note: with limit_bounds=True the top level has the size of the bounds rectangle,
    not openslide_object.dimensions.
    """
    return (
        tuple(slide.tiles.level_tiles[-1]),
        tuple(slide.tiles.level_dimensions[-1]),
    )


def assert_same_grid(slides):
    """Raise ValueError unless all slides share the same top-level grid and size.

    Returns the common ((cols, rows), (width, height)).
    """
    if not slides:
        raise ValueError("No slides given.")
    ref_grid, ref_size = get_deepzoom_grid(slides[0])
    for s in slides[1:]:
        grid, size = get_deepzoom_grid(s)
        if grid != ref_grid or size != ref_size:
            raise ValueError(
                f"DeepZoom grid mismatch: {slides[0].name} has grid {ref_grid} / "
                f"size {ref_size}, {s.name} has grid {grid} / size {size}."
            )
    return ref_grid, ref_size


def _raise_open_file_limit(needed):
    """Best effort: raise the soft open-file limit so a lazy arrayjoin can keep
    one handle per tile open. No-op on platforms without `resource`."""
    try:
        # Standard Library
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = needed + 256
        if soft != resource.RLIM_INFINITY and soft < target:
            new_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ImportError, ValueError, OSError):
        pass


def stitch_tiles(
    tile_dir,
    out_file,
    grid,
    full_size,
    *,
    tile_size=DEFAULT_TILE_SIZE,
    background=DEFAULT_BACKGROUND,
    bands=3,
    logger=None,
    progress=True,
    desc="Reconstructing slide",
    thumbnail_path=None,
    thumbnail_size=512,
):
    """Stitch '{col}_{row}.tif' tiles from `tile_dir` into a pyramidal TIFF.

    Args:
        tile_dir (str): Directory containing the tiles.
        out_file (str): Output file path (parent directory is created).
        grid (tuple): (cols, rows) of the tile grid.
        full_size (tuple): (width, height) in pixels the result is cropped to.
        tile_size (int): Edge length of a full grid cell.
        background (int): Grey value used for missing tiles.
        bands (int): 3 for RGB output, 1 for single-channel (e.g. masks).
        logger (logging.Logger, optional): Logger.
        progress (bool): Show a tqdm progress bar.
        desc (str): Progress bar label.
        thumbnail_path (str, optional): If given, a PNG thumbnail of the result is
            written there. A thumbnail failure is logged and does not raise.
        thumbnail_size (int): Longest edge of the thumbnail in pixels.

    Returns:
        int: Number of tiles found on disk.

    Raises:
        ValueError: On a missing directory, no tiles found, or invalid arguments.
    """
    # Third Party
    import pyvips

    log = logger or logging.getLogger(__name__)

    if bands not in (1, 3):
        raise ValueError("bands must be 1 or 3.")
    if not os.path.isdir(tile_dir):
        raise ValueError(f"Input path {tile_dir} does not exist.")
    cols, rows = grid
    width, height = full_size
    if cols < 1 or rows < 1:
        raise ValueError(f"Invalid grid {grid}.")

    start = time()
    _raise_open_file_limit(cols * rows)

    fill = [background] * bands
    placeholder = (
        pyvips.Image.black(tile_size, tile_size, bands=bands)
        .new_from_image(fill)
        .cast("uchar")
    )

    tiles, found = [], 0
    for row in tqdm(range(rows), desc=desc, disable=not progress):
        for col in range(cols):
            path = os.path.join(tile_dir, f"{col}_{row}.tif")
            if not os.path.exists(path):
                tiles.append(placeholder)
                continue
            img = pyvips.Image.new_from_file(path)
            if img.bands == 4:
                img = img.flatten(background=[255, 255, 255])
            if bands == 3 and img.bands == 1:
                img = img.bandjoin([img, img])
            elif bands == 1 and img.bands > 1:
                img = img.extract_band(0)
            tiles.append(img.cast("uchar"))
            found += 1

    if found == 0:
        raise ValueError(f"No tiles named '{{col}}_{{row}}.tif' found in {tile_dir}.")

    interpretation = "srgb" if bands == 3 else "b-w"
    wsi = pyvips.Image.arrayjoin(tiles, across=cols, background=fill).copy(
        interpretation=interpretation
    )
    wsi = wsi.crop(0, 0, min(width, wsi.width), min(height, wsi.height))

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    log.info(f"Saving reconstructed slide to {out_file}")
    t_save = time()
    wsi.tiffsave(
        out_file,
        tile=True,
        compression="jpeg",
        bigtiff=True,
        pyramid=True,
        tile_width=256,
        tile_height=256,
    )
    log.info(
        f"Reconstructed {found}/{rows * cols} tiles; total "
        f"{round((time() - start) / 60, 2)} min "
        f"(save: {round((time() - t_save) / 60, 2)} min)."
    )
    if thumbnail_path is not None:
        save_thumbnail(out_file, thumbnail_path, size=thumbnail_size, logger=log)
    return found


def save_thumbnail(wsi_path, output_path, size=512, logger=None):
    """Create and save a PNG thumbnail of a (pyramidal) image.

    Args:
        wsi_path (str): Image the thumbnail is created from.
        output_path (str): Where the PNG is written.
        size (int): Bounding box edge length; aspect ratio is preserved.
        logger (logging.Logger, optional): Logger.

    Returns:
        bool: True if the thumbnail was written.
    """
    log = logger or logging.getLogger(__name__)
    log.info("Creating PNG thumbnail for reconstructed WSI")
    try:
        thumb = pyvips.Image.thumbnail(wsi_path, size, height=size)
        thumb.write_to_file(output_path)
        log.debug(f"PNG thumbnail saved to {output_path}")
        return True
    except Exception as e:
        log.error(f"Error saving PNG thumbnail: {e}")
        return False
