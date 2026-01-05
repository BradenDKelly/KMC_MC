# Equation of State (EOS) Implementations

## LJTS Thol 2014 EOS

**IMPORTANT**: The implementation in `ljts_thol_2014.py` is currently a **PLACEHOLDER**.

The actual Thol et al. 2014 EOS requires coefficients from the paper:
- Thol, M., et al. "Equation of State for the Lennard-Jones Truncated and Shifted Model Fluid with a Cutoff Radius of 2.5σ" (2014)

To complete the implementation:
1. Obtain the coefficients/parameters from the Thol 2014 paper
2. Replace the placeholder functions (`a_res`, `u_res`, `Z`, `P`, `mu_res`) with the actual EOS equations
3. The EOS should be in reduced units (ε = σ = kB = 1) for cutoff rc = 2.5σ

The test `tests/test_ljts_eos_consistency.py` is currently skipped until the EOS is properly implemented.

