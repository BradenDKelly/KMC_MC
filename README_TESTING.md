# Testing Guide

## Installation

### Base Requirements

Install base dependencies:
```bash
pip install -r requirements.txt
```

This installs:
- `numpy>=1.20.0` - Numerical computing
- `numba>=0.56.0` - JIT compilation for performance-critical code

### Development Dependencies

For development and testing:
```bash
pip install -r requirements-dev.txt
```

This includes:
- All base requirements
- `pytest>=7.0.0` - Testing framework
- `pytest-cov>=4.0.0` - Test coverage reporting

### Quick Start

```bash
# Install all dependencies
pip install -r requirements-dev.txt

# Run fast tests (recommended for development)
py -m pytest -m "not slow" tests/

# Run all tests (including slow ones)
py -m pytest tests/ --runslow
```

## Performance Requirements

### Numba Installation (Required)

**Performance-critical code paths require Numba to be installed.**

Numba is included in `requirements.txt` and `requirements-dev.txt`. If installing manually:
```bash
pip install numba
```

Performance-critical code includes:
- MC/kMC samplers
- Energy calculations in simulation loops
- Virial pressure calculations
- Widom insertion calculations

If numba is not installed, performance tests will fail immediately with a clear error message telling you to install it.

**Note for developers:** Lightweight unit tests that validate math/geometry (without running sweeps) may work without numba if you set the environment variable `KMC_ALLOW_PYTHON=1`. However, this is intended for debugging only - the default behavior is strict (numba required).

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

## Numba as Required Backend

Numba-accelerated kernels are the **primary execution path** for performance-critical code:
- `total_energy_numba`: Fast total energy calculation
- `virial_pressure_numba`: Fast virial pressure calculation
- `delta_energy_particle_move_numba`: Fast local energy changes

These kernels are tested for consistency with Python reference implementations in `tests/test_lj_numba_consistency.py`.

**Numba is required** - there is no automatic fallback to Python for performance-critical code paths. This ensures consistent performance characteristics across all simulation runs.

## Running Specific Tests

```bash
# Run only EOS consistency tests
py -m pytest tests/test_ljts_eos_consistency.py

# Run only fast tests
py -m pytest -m "not slow" tests/

# Run with verbose output
py -m pytest -v tests/
```

