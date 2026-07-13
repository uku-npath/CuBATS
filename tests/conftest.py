"""Shared pytest fixtures and setup for the CuBATS test suite."""
# Third Party
import openslide.lowlevel as _ll

_original_read_icc_profile = _ll.read_icc_profile


def _safe_read_icc_profile(osr):
    """
    Guard against openslide-python >=1.4 crashing with
    'ValueError: Array length must be >= 0, not -1' when the underlying
    OpenSlide backend (e.g. generic TIFF, used by our test fixtures)
    doesn't support ICC profile size reporting.
    """
    try:
        return _original_read_icc_profile(osr)
    except ValueError:
        return None


_safe_read_icc_profile.available = getattr(_original_read_icc_profile, "available", True)

_ll.read_icc_profile = _safe_read_icc_profile
