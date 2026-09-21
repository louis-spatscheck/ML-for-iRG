"""Critical values of the 2D Ising model used in the analysis scripts."""
import math

# Exact critical inverse temperature (Onsager): beta_c = ln(1 + sqrt(2)) / 2 = 0.440686...
BETA_C_EXACT = 0.5 * math.log(1 + math.sqrt(2))

# Estimated critical inverse temperatures at which the critical exponents are
# extracted. They differ slightly from the exact value because they are
# estimated from the simulation data.
#
# Forward RG analysis (scripts/analyze_RG.py). The thesis text does not state
# where this value comes from.
BETA_C_FORWARD = 0.44048
# Inverse RG analysis (scripts/analyze_inverseRG.py). Thesis Sec. 8.1 reports
# beta_c = 0.44(047), estimated from the comparison of the L = 32 systems.
BETA_C_INVERSE = 0.4404728

# Exact ratios of critical exponents (2D Ising):
# beta_m / nu (magnetization) and gamma / nu (susceptibility).
BETA_OVER_NU = 1 / 8
GAMMA_OVER_NU = 7 / 4
