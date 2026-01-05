"""Performance backend selection and requirements.

This module centralizes the decision to use Numba-accelerated kernels
for performance-critical code paths (samplers, energy calculations, etc.).

Numba is required for all simulation-style code. Python reference implementations
are kept only for correctness testing, not for production runs.
"""

import os

# Try to import numba
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    njit = None


def require_numba(feature: str):
    """Require Numba to be available, raising a clear error if not.
    
    Args:
        feature: Description of the feature that requires numba (e.g., "MC/kMC samplers")
        
    Raises:
        ImportError: If numba is not available and fallback is not allowed
    """
    if NUMBA_AVAILABLE:
        return
    
    # Check for escape hatch environment variable
    allow_python = os.getenv("KMC_ALLOW_PYTHON", "0").lower() in ("1", "true", "yes")
    if allow_python:
        # Escape hatch: allow fallback (for tiny unit tests only)
        return
    
    # Default: strict requirement
    raise ImportError(
        f"Numba is required for {feature}. "
        f"Install with: pip install numba\n"
        f"(To allow Python fallback for unit tests only, set KMC_ALLOW_PYTHON=1)"
    )


# Export backend status
__all__ = ["NUMBA_AVAILABLE", "require_numba", "njit"]

