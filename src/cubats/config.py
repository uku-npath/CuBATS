# config.py
# Standard Library
import platform

# Third Party
from array_api_compat import get_namespace

gpu_enabled = False


def get_backend_namespace():
    """
    Detects the appropriate backend (NumPy or CuPy) and returns the array API namespace.

    Returns:
        module: The array API namespace (NumPy or CuPy).
    """
    global gpu_enabled
    try:
        if platform.system() in ("Windows", "Linux"):
            # Third Party
            import cupy as cp  # type: ignore

            if cp.cuda.runtime.getDeviceCount() > 0:
                gpu_enabled = True
                # print("Using CuPy for GPU acceleration.")
                return get_namespace(cp.array([0]))  # Return CuPy namespace
    except Exception:
        pass

    # Fallback to NumPy
    gpu_enabled = False
    # Third Party
    import numpy as np

    # print("Using NumPy.")
    return get_namespace(np.array([0]))  # Return NumPy namespace


def get_backend_info():
    """
    Returns a string indicating the backend being used (NumPy or CuPy).

    Returns:
        str: "CuPy" if CuPy is being used, otherwise "NumPy".
    """
    return "CuPy (GPU)" if gpu_enabled else "NumPy (CPU)"


# Single call to get the namespace
xp = get_backend_namespace()
