# Standard Library
import functools
import logging.config
import multiprocessing as mp
import os
import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from os import listdir, path
from time import time
from typing import List, Optional, Tuple, Union

# Third Party
import numpy as np
import onnx
import torch
import torch.nn.functional as F
import torchstain
import torchvision
from onnx2torch import convert
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
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


DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 4


def run_tumor_segmentation(
    input_path: str,
    model_path: str,
    tile_size: Tuple[int, int],
    output_path: Union[str, None] = None,
    normalization: bool = False,
    inversion: bool = False,
    plot_results: bool = False,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
):
    """Run the segmentation pipeline on the given input path using the specified model.

    Performs segmentation on a single HE stained WSI or all HE stained WSIs in a directory using the specified model.
    The segmentation results are saved as .TIFF in the output directory, as well as a .PNG thumbnail. If no output
    directory is provided, the results are saved in the same directory as the input. Optionally, a thumbnail of the
    segmentation results can be plotted on the original image and saved.

    When available tiles are processed in GPU batches. When multiple GPUs are available and multiple
    files are being processed, files are distributed across GPUs for concurrent processing.

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
        batch_size (Optional[int], optional): Number of tiles per GPU forward pass. If None, this is auto-tuned per
            device by probing increasing batch sizes against the model until it is close to exhausting GPU memory,
            then backing off with a safety margin. On CPU, a small fixed default is used instead. Defaults to None.
        num_workers (Optional[int], optional): Number of background threads used to prefetch/preprocess tiles (tile
            decoding, normalization, resizing). If None, this is derived from the CPU core count. Defaults to None.
        gpu_ids (Optional[List[int]], optional): Which CUDA device indices to use. If None, all visible GPUs are
            used. Ignored if CUDA is unavailable (falls back to CPU). Defaults to None.

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
        f"inversion: {inversion}, plot_results: {plot_results}, batch_size: {batch_size}, "
        f"num_workers: {num_workers}, gpu_ids: {gpu_ids}"
    )

    start_time_segmentation = time()

    # Check if the input path is valid and if it is a file or a directory
    try:
        if not path.exists(input_path):
            raise FileNotFoundError(f"Input path {input_path} does not exist.")
        if path.isfile(input_path):
            segment_single_file = True
            input_folder = path.dirname(input_path)
        else:
            segment_single_file = False
            input_folder = input_path
    except FileNotFoundError as e:
        logger.error(e)
        raise
    except Exception as e:
        logger.error(f"Unexpected error checking input path: {e}")
        raise

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

    # Check if the model is a valid onnx file
    try:
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
    except ValueError as e:
        logger.error(f"Model {model_path} is not a valid onnx file.")
        raise e
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}")
        raise ValueError(f"Model {model_path} is not a valid path.")

    try:
        model_input_size = _get_model_input_size(onnx_model)
    except Exception as e:
        logger.error(f"Error getting model input size: {e}")
        raise RuntimeError(f"Error getting model input size: {e}")

    # Suppress Pillow warning because of the large WSI sizes.
    Image.MAX_IMAGE_PIXELS = None

    devices = _resolve_devices(gpu_ids)
    logger.info(f"Using device(s): {[str(d) for d in devices]}")

    # Run tumor segmentation
    if segment_single_file:
        logger.info("Segmentation-mode: single file")
        model = _load_model_on_device(onnx_model, devices[0])
        resolved_batch_size = batch_size if batch_size is not None else _auto_batch_size(
            model, devices[0], model_input_size
        )
        resolved_num_workers = num_workers if num_workers is not None else _auto_num_workers()
        logger.info(
            f"Using batch_size={resolved_batch_size}, num_workers={resolved_num_workers}"
        )
        _segment_file(
            input_path,
            model,
            tile_size,
            model_input_size,
            output_path,
            normalization,
            inversion,
            plot_results,
            device=devices[0],
            batch_size=resolved_batch_size,
            num_workers=resolved_num_workers,
        )
    else:
        logger.info("Segmentation-mode: multiple files")
        files = [
            path.join(input_folder, f) for f in listdir(input_folder) if f.endswith(".tif")
        ]
        if len(devices) > 1 and devices[0].type == "cuda":
            _segment_files_multi_gpu(
                files,
                model_path,
                tile_size,
                output_path,
                normalization,
                inversion,
                plot_results,
                batch_size,
                num_workers,
                devices,
            )
        else:
            model = _load_model_on_device(onnx_model, devices[0])
            resolved_batch_size = batch_size if batch_size is not None else _auto_batch_size(
                model, devices[0], model_input_size
            )
            resolved_num_workers = num_workers if num_workers is not None else _auto_num_workers()
            logger.info(
                f"Using batch_size={resolved_batch_size}, num_workers={resolved_num_workers}"
            )
            for file in files:
                _segment_file(
                    file,
                    model,
                    tile_size,
                    model_input_size,
                    output_path,
                    normalization,
                    inversion,
                    plot_results,
                    device=devices[0],
                    batch_size=resolved_batch_size,
                    num_workers=resolved_num_workers,
                )
        logger.info(f"Segmentation of all files in {input_folder} completed.")

    end_time_segmentation = time()
    logger.info(
        f"Segmentation of {input_path} completed in"
        f"{(end_time_segmentation - start_time_segmentation) / 60: .2f} Minutes."
    )


def _segment_files_multi_gpu(
    files: List[str],
    model_path: str,
    tile_size,
    output_path,
    normalization,
    inversion,
    plot_results,
    batch_size,
    num_workers,
    devices: List[torch.device],
):
    """Distribute a list of WSI files across multiple GPU worker processes for segmentation.

    Each worker process is pinned to a single GPU, loads its own copy of the converted model once, and then pulls
    files off a shared queue.
    """
    ctx = mp.get_context("spawn")  # required for safe CUDA usage across processes
    task_queue = ctx.Queue()
    for f in files:
        task_queue.put(f)

    processes = []
    for device in devices:
        p = ctx.Process(
            target=_gpu_worker_loop,
            args=(
                device.index,
                model_path,
                task_queue,
                tile_size,
                output_path,
                normalization,
                inversion,
                plot_results,
                batch_size,
                num_workers,
                len(devices),
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def _gpu_worker_loop(
    gpu_id,
    model_path,
    task_queue,
    tile_size,
    output_path,
    normalization,
    inversion,
    plot_results,
    batch_size,
    num_workers,
    num_concurrent_gpu_workers=1,
):
    """Worker entry point: load the model once on the assigned GPU, then process files from the queue."""
    device = torch.device(f"cuda:{gpu_id}")
    try:
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        model_input_size = _get_model_input_size(onnx_model)
        model = _load_model_on_device(onnx_model, device)
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Failed to initialize model: {e}")
        return

    resolved_batch_size = batch_size if batch_size is not None else _auto_batch_size(
        model, device, model_input_size
    )
    resolved_num_workers = (
        num_workers
        if num_workers is not None
        else _auto_num_workers(num_concurrent_gpu_workers)
    )
    logger.info(
        f"[GPU {gpu_id}] Using batch_size={resolved_batch_size}, num_workers={resolved_num_workers}"
    )

    while True:
        try:
            file_path = task_queue.get_nowait()
        except queue.Empty:
            break
        logger.info(f"[GPU {gpu_id}] Segmenting {file_path}")
        try:
            _segment_file(
                file_path,
                model,
                tile_size,
                model_input_size,
                output_path,
                normalization,
                inversion,
                plot_results,
                device=device,
                batch_size=resolved_batch_size,
                num_workers=resolved_num_workers,
            )
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] Failed on {file_path}: {e}")


def _segment_file(
    file_path,
    model,
    tile_size,
    model_input_size,
    output_path,
    normalization,
    inversion,
    plot_results,
    device: torch.device = torch.device("cpu"),
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
):
    """Performs tumor detection and segmentation on a single WSI file.

    Tiles are prefetched and preprocessed concurrently in a background thread pool while the main loop batches
    preprocessed tiles and runs them through the model in a single forward pass per batch. Output tiles are collected
    and assembled into a single full-slide array at the end, then wrapped in one pyvips.Image for saving.

    Args:
        file_path (str): The path to the WSI file.
        model: The segmentation model, already moved to `device` and in eval mode.
        tile_size (tuple): The size of each tile in pixels.
        model_input_size (tuple): The input size of the model.
        output_path (str): The path to save the segmented image.
        normalization (bool): Whether to normalize the input tiles.
        inversion (bool): Whether to invert the segmentation output.
        plot_results (bool): Whether to plot the segmentation results on the original image.
        device (torch.device): The device the model lives on and batches should be moved to.
        batch_size (int): Number of tiles per GPU forward pass.
        num_workers (int): Number of background threads prefetching/preprocessing tiles.

    Returns:
        None
    """
    logger.info(f"Starting segmentation for file: {file_path} on {device}")
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
    normalizer = None
    if normalization:
        ref_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "assets", "ref_norm.png")
        )
        normalizer = _init_normalizer(ref_path)

    n_cols, n_rows = slide_generator.level_tiles[-1]
    level = slide_generator.level_count - 1

    # Whether the rightmost column / bottom row is a partial (truncated) tile that needs white-padding.
    has_right_overhang = (n_cols * tile_size[0] - slide.dimensions[0]) > 0
    has_bottom_overhang = (n_rows * tile_size[1] - slide.dimensions[1]) > 0

    needs_resize = tile_size != model_input_size[2:]

    coords = [(row, col) for row in range(n_rows) for col in range(n_cols)]
    output_tiles = [None] * len(coords)
    use_amp = device.type == "cuda"

    preprocess_fn = functools.partial(
        _preprocess_tile,
        slide_generator=slide_generator,
        level=level,
        tile_size=tile_size,
        n_cols=n_cols,
        n_rows=n_rows,
        has_right_overhang=has_right_overhang,
        has_bottom_overhang=has_bottom_overhang,
        normalization=normalization,
        normalizer=normalizer,
        needs_resize=needs_resize,
        model_input_size=model_input_size,
    )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        pending = _bounded_imap(executor, preprocess_fn, coords, max_pending=num_workers * 4)

        pbar = tqdm(
            total=len(coords),
            desc=f"Segmenting {path.splitext(path.basename(file_path))[0]}",
        )

        batch_coords: List[Tuple[int, int]] = []
        batch_tensors: List[torch.Tensor] = []

        for coord, tensor in pending:
            batch_coords.append(coord)
            batch_tensors.append(tensor)
            if len(batch_tensors) >= batch_size:
                _run_batch(
                    batch_coords, batch_tensors, model, device, use_amp,
                    needs_resize, tile_size, inversion, output_tiles, n_cols, pbar,
                )
                batch_coords.clear()
                batch_tensors.clear()

        _run_batch(
            batch_coords, batch_tensors, model, device, use_amp,
            needs_resize, tile_size, inversion, output_tiles, n_cols, pbar,
        )  # remaining partial batch
        pbar.close()

    logger.debug("Assembling segmented WSI into a single contiguous array")
    row_arrays = [
        np.concatenate(output_tiles[row * n_cols:(row + 1) * n_cols], axis=1)
        for row in range(n_rows)
    ]
    segmented_wsi_array = np.concatenate(row_arrays, axis=0)
    segmented_wsi = VipsImage.new_from_array(segmented_wsi_array).copy(interpretation="b-w")

    # Extract the base name and file type
    base_name, file_type = path.splitext(file_path)
    wsi_name = cutils.get_name(base_name)

    # Construct output paths for the mask and thumbnail
    mask_out = path.join(output_path, f"{wsi_name}_mask{file_type}")
    thumb_out = path.join(output_path, f"{wsi_name}_mask_thumbnail.png")

    # Save the segmented WSI using pyvips.
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


def _preprocess_tile(
    coord,
    slide_generator,
    level,
    tile_size,
    n_cols,
    n_rows,
    has_right_overhang,
    has_bottom_overhang,
    normalization,
    normalizer,
    needs_resize,
    model_input_size,
):
    """Fetch and preprocess a single tile. Runs in a background worker thread."""
    row, col = coord
    needs_padding = (col == n_cols - 1 and has_right_overhang) or (
        row == n_rows - 1 and has_bottom_overhang
    )

    raw_tile = slide_generator.get_tile(level, (col, row))
    if needs_padding:
        tile = Image.new("RGB", tile_size, (255, 255, 255))
        tile.paste(raw_tile, (0, 0))
    else:
        tile = raw_tile

    transform = torchvision.transforms.ToTensor()
    tile_tensor = transform(tile).float()  # [0, 1] scale

    if normalization:
        # Reinhard normalizer expects [0, 255]-scale input.
        tile_tensor = tile_tensor * 255
        normalized_tile = normalizer.normalize(tile_tensor)
        tile_tensor = normalized_tile.cpu().permute(2, 0, 1).float()
        if needs_resize:
            tile_tensor = F.interpolate(
                tile_tensor.unsqueeze(0),
                size=model_input_size[2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
    else:
        if needs_resize:
            rescaled_tile = resize(
                np.asarray(tile),
                (model_input_size[2], model_input_size[3]),
                anti_aliasing=True,
            )
            tile_tensor = (
                torch.from_numpy(rescaled_tile).type(torch.float32).permute(2, 0, 1)
            )
        else:
            tile_tensor = tile_tensor * 255

    return coord, tile_tensor


def _bounded_imap(executor: ThreadPoolExecutor, fn, items, max_pending: int):
    """Like executor.map, but keeps at most `max_pending` futures in flight.

    This bounds memory usage (preprocessed tiles waiting for the GPU) regardless of how many tiles a slide has.
    """
    items = iter(items)
    futures = deque()

    for _ in range(max_pending):
        try:
            futures.append(executor.submit(fn, next(items)))
        except StopIteration:
            break

    while futures:
        fut = futures.popleft()
        yield fut.result()
        try:
            futures.append(executor.submit(fn, next(items)))
        except StopIteration:
            continue


def _run_batch(
    batch_coords,
    batch_tensors,
    model,
    device: torch.device,
    use_amp: bool,
    needs_resize: bool,
    tile_size,
    inversion: bool,
    output_tiles: list,
    n_cols: int,
    pbar,
):
    """Run one pending batch of preprocessed tiles through the model and write thresholded
    results into output_tiles.
    """
    if not batch_tensors:
        return

    batch = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)

    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        segmentation = model(batch)

    segmentation = segmentation.sigmoid()
    if segmentation.dim() == 4 and segmentation.shape[1] == 1:
        segmentation = segmentation.squeeze(1)

    if needs_resize:
        # Resize the whole batch back to tile_size in one call, instead of per-tile on CPU.
        segmentation = F.interpolate(
            segmentation.unsqueeze(1),
            size=tile_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    segmentation = segmentation.cpu().numpy()

    for i, coord in enumerate(batch_coords):
        seg = segmentation[i]
        if inversion:
            seg = 1 - seg
        seg = (seg > 0.5).astype(np.uint8) * 255

        row, col = coord
        output_tiles[row * n_cols + col] = seg

    pbar.update(len(batch_coords))


def _save_segmented_wsi(segmented_wsi, tile_size, output_path):
    """Save the segmented WSI to file using CCITT Group 4 (fax-style) compression.

    Args:
        segmented_wsi (pyvips.Image): The segmented WSI image.
        tile_size (tuple): Tile width/height to use for the saved pyramid.
        output_path (str): The path to save the segmented WSI.
    """
    logger.info(f"Saving segmented WSI to {output_path}")
    try:
        segmented_wsi.crop(0, 0, segmented_wsi.width, segmented_wsi.height).tiffsave(
            output_path,
            tile=True,
            compression="ccittfax4",
            squash=True,  # pack the 8-bit b/w image (values 0/255) down to true 1-bit
            bigtiff=True,
            pyramid=True,
            tile_width=tile_size[0],
            tile_height=tile_size[1],
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


def _get_model_input_size(model):
    """
    Get the expected input size for the model.

    Args:
        model: The onnx model.

    Returns:
        tuple: The expected input size for the model in the format (1, C, H, W).
    """
    input_tensor = model.graph.input[0]
    input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    # Ensure the batch size is set to 1
    input_shape[0] = 1
    logger.debug(f"Model input size: {input_shape}")
    return tuple(input_shape)


def _resolve_devices(gpu_ids: Optional[List[int]]) -> List[torch.device]:
    """Resolve which torch devices to run on.

    Args:
        gpu_ids: Requested CUDA device indices, or None to use all visible GPUs.

    Returns:
        List[torch.device]: One or more CUDA devices, or a single CPU device if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - running on CPU.")
        return [torch.device("cpu")]

    available = list(range(torch.cuda.device_count()))
    if gpu_ids is None:
        gpu_ids = available
    else:
        invalid = [g for g in gpu_ids if g not in available]
        if invalid:
            raise ValueError(f"Requested GPU id(s) {invalid} not available. Available: {available}")

    return [torch.device(f"cuda:{i}") for i in gpu_ids]


def _load_model_on_device(onnx_model, device: torch.device):
    """Convert an onnx model to torch and move it to the given device."""
    try:
        model = convert(onnx_model)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error converting/moving model to {device}: {e}")
        raise RuntimeError(f"Error converting/moving model to {device}: {e}")


def _auto_num_workers(num_concurrent_gpu_workers: int = 1, cap: int = 16) -> int:
    """Pick a number of tile-prefetch threads based on available CPU cores.

    When multiple GPU worker processes run concurrently, each spawns its own thread pool, so
    the data is divided across them to avoid oversubscribing.

    Args:
        num_concurrent_gpu_workers: Number of GPU worker processes to be running at once.
        cap: Upper bound on threads per worker, to avoid over-subscribing.

    Returns:
        int: Number of worker threads to use.
    """
    cpu_count = os.cpu_count() or 4
    per_worker = max(1, cpu_count // max(1, num_concurrent_gpu_workers))
    return min(cap, per_worker)


def _auto_batch_size(
    model,
    device: torch.device,
    model_input_size,
    max_batch_size: int = 1024,
    safety_factor: float = 0.8,
) -> int:
    """Auto-tune the GPU batch size by probing increasing batch sizes until memory runs out.

    Doubles the batch size starting from 1 until a forward pass raises an out-of-memory error (or max_batch_size is
    reached), then backs off to a fraction of the largest size that succeeded to leave headroom for
    the CPU->GPU transfer of the next batch. On CPU, this returns a small fixed default instead.

    Args:
        model: The model already moved to `device`, in eval mode.
        device: The device to probe.
        model_input_size: (1, C, H, W) tuple describing the model's expected input shape.
        max_batch_size: Upper bound to stop probing at, even if memory would allow more.
        safety_factor: Fraction of the largest successful batch size to actually use.

    Returns:
        int: Batch size.
    """
    if device.type != "cuda":
        return 4

    _, channels, height, width = model_input_size

    last_success = 1
    candidate = 1
    while candidate <= max_batch_size:
        dummy = None
        try:
            torch.cuda.empty_cache()
            dummy = torch.zeros((candidate, channels, height, width), device=device)
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=True):
                _ = model(dummy)
            torch.cuda.synchronize(device)
            last_success = candidate
            candidate *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            raise
        finally:
            del dummy
            torch.cuda.empty_cache()

    safe_batch_size = max(1, int(last_success * safety_factor))
    logger.info(
        f"Auto-tuned batch size on {device}: probed up to {last_success}, "
        f"using {safe_batch_size} (safety_factor={safety_factor})"
    )
    return safe_batch_size


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
