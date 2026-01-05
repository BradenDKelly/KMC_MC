import numpy as np
import pytest

from src.neighborlist_numba import minimum_image_scalar


def test_numba_floor_smoke():
    # compile and run minimum_image_scalar
    dr = np.array([1.2, -2.7, 0.49], dtype=np.float64)
    out = minimum_image_scalar(dr, 5.0)
    assert out.shape == (3,)


def test_numba_simple_floor_inline():
    # compile a trivial njit that uses np.floor
    from numba import njit

    @njit(cache=False)
    def f(x):
        return np.floor(x)

    assert float(f(1.7)) == 1.0


def test_build_neighbor_list_numba_compiles():
    from src.neighborlist_numba import build_neighbor_list_numba
    
    rng = np.random.default_rng(0)
    N = 32
    L = 5.0
    positions = rng.random((N,3))
    positions *= L
    rc2 = (2.5 + 0.2)**2
    nl, starts = build_neighbor_list_numba(positions, L, rc2)
    assert starts.shape == (N+1,)
    assert nl.ndim == 1

