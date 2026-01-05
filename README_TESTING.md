# Testing Guide

## Fast vs Full Test Runs

### Fast Run (default, excludes slow tests)
```bash
py -m pytest tests/ -m "not slow"
```
Runs all tests except those marked as `@pytest.mark.slow`. Recommended for quick validation during development.

**Target runtime: <10s**

### Full Run (includes slow tests)
```bash
py -m pytest tests/
```
Runs all tests including slow ones. Use for comprehensive validation.

**Expected runtime: ~30-70s (depending on numba availability)**

### Alternative: Use --runslow flag
```bash
py -m pytest tests/ --runslow
```
Same as full run - runs all tests including slow ones.

## Test Categories

### Fast Tests
- Unit tests for individual modules
- Small system regression tests (N=32)
- MC vs kMC consistency checks

### Slow Tests
- Large system EOS validation (N=108)
- Tests marked with `@pytest.mark.slow`

## Numba Acceleration

If `numba` is installed, certain hot loops are automatically accelerated:
- `total_energy_numba`: Fast total energy calculation
- `virial_pressure_numba`: Fast virial pressure calculation
- `delta_energy_particle_move_numba`: Fast local energy changes

Numba-accelerated functions are tested for consistency with Python reference implementations in `tests/test_lj_numba_consistency.py`.

If numba is not available, the code gracefully falls back to pure Python implementations.

## Running Specific Tests

```bash
# Run only EOS consistency tests
py -m pytest tests/test_ljts_eos_consistency.py

# Run only fast tests
py -m pytest -m "not slow" tests/

# Run with verbose output
py -m pytest -v tests/
```

