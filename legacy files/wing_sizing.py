import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from constants import G, RHO_ORIGIN, V_CRUISE, hcruise, number_of_wings, area_split, wing_span

# ============================================================
# 1. INPUTS FROM MATCHING DIAGRAM AND WEIGHT ESTIMATION
# ============================================================

MTOW =  2000                      # [kg] Class I / Class II mass estimate |  CHANGE THIS TO IMPORT MTOW
W_S = 760                       # [N/m^2] selected wing loading from matching diagram | CHANGE THIS TO IMPORT W/S

# For cruise check
V_cruise = V_CRUISE               # [m/s] cruise speed

# For stall check
rho_stall = RHO_ORIGIN            # [kg/m^3] usually sea-level density


# ============================================================
# 2. CONFIGURATION PARAMETERS
# ============================================================
number_of_wings     = number_of_wings     # [-] e.g. 1 for conventional, 2 for Prandtl/box-wing
area_split          = area_split          # [-] fraction of total area assigned to one wing
wing_span           = wing_span           # [-] span from the footprint constraint




# ============================================================
# 3. CHOSEN PARAMETERS
# ============================================================

CL_max = 2.0                   # [-] assumed maximum lift coefficient 
taper_ratio = 0.8              # [-] lambda = c_tip / c_root

# ============================================================
# 4. CRUISE DENSITY
# ============================================================

def compute_isa_density(altitude_m):
    temperature_k = 288.15 - 0.0065 * altitude_m
    pressure_pa = 101325.0 * (temperature_k / 288.15) ** 5.25588
    return pressure_pa / (287.05 * temperature_k)

altitude = 3000 # temporary
# CHANGE TO hcruise IN BRACKETS when hcruise DEFINED IN @constants.py
rho_cruise = compute_isa_density(altitude)  # [kg/m^3] air density at cruise altitude

# ============================================================
# 5. WING-RELATED PARAMETERS CALCULATIONS
# ============================================================

def compute_weight(mtow_kg):
    return mtow_kg * G


def compute_total_wing_area(weight_n, wing_loading_n_m2):
    return weight_n / wing_loading_n_m2


def compute_area_per_wing(total_area_m2, split_fraction):
    return total_area_m2 * split_fraction


def compute_aspect_ratio(total_area_m2, span_m):
    return span_m ** 2 / total_area_m2


def compute_average_chords(total_area_m2, area_per_wing_m2, span_m):
    c_avg_total = total_area_m2 / span_m
    c_avg_wing = area_per_wing_m2 / span_m
    return c_avg_total, c_avg_wing


def compute_root_tip_chords(area_per_wing_m2, span_m, taper):
    c_root = (2.0 * area_per_wing_m2) / (span_m * (1.0 + taper))
    c_tip = taper * c_root
    return c_root, c_tip


def compute_cruise_lift_coefficient(weight_n, density_kg_m3, speed_m_s, total_area_m2):
    q_cruise = 0.5 * density_kg_m3 * speed_m_s ** 2
    return weight_n / (q_cruise * total_area_m2)


def compute_stall_speed(weight_n, density_kg_m3, total_area_m2, cl_max):
    return math.sqrt((2.0 * weight_n) / (density_kg_m3 * total_area_m2 * cl_max))


def main():
    W = compute_weight(MTOW)
    S_total = compute_total_wing_area(W, W_S)
    S_wing = compute_area_per_wing(S_total, area_split)

    b = wing_span
    AR = compute_aspect_ratio(S_total, b)
    c_avg_total, c_avg_wing = compute_average_chords(S_total, S_wing, b)
    c_root, c_tip = compute_root_tip_chords(S_wing, b, taper_ratio)

    CL_cruise = compute_cruise_lift_coefficient(W, rho_cruise, V_CRUISE, S_total)
    V_stall = compute_stall_speed(W, RHO_ORIGIN, S_total, CL_max)

    print(f"Aircraft weight W = {W:.2f} N")
    print(f"Total reference wing area S = {S_total:.2f} m^2")
    print(f"Number of wings = {number_of_wings}")
    print(f"Area per wing = {S_wing:.2f} m^2")
    print(f"Aspect ratio AR = {AR:.2f}")
    print(f"Span b = {b:.2f} m")
    print(f"Average chord based on total area = {c_avg_total:.2f} m")
    print(f"Average chord per wing = {c_avg_wing:.2f} m")
    print(f"Root chord = {c_root:.2f} m")
    print(f"Tip chord = {c_tip:.2f} m")
    print(f"Cruise lift coefficient CL_cruise = {CL_cruise:.3f}")
    print(f"Stall speed V_stall = {V_stall:.2f} m/s")


if __name__ == "__main__":
    main()
