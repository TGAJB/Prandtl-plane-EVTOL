"""
payload_range.py
================
Payload-range diagram for the fixed-battery Folding Prandtl eVTOL.


battery mass is fixed

- Installed pack energy and the 20% reserve are fixed (from battery sizing).
- The vertical phases + climb + descent are treated as a fixed "mission
  overhead" energy that every flight pays regardless of range; only the
  CRUISE leg extends with range.
- Cruise power scales with mass via energy.py's cruise_power(mtow), so a
  lighter (less payload) aircraft cruises on less power and reaches further.
- The curve is ANCHORED at the design point (4 pax -> 200 km) so it is a
  trade study around the design mission, not an absolute-range prediction.

Range extension is therefore robust as a TREND (~slope, km per passenger);
absolute values away from the anchor are first-order estimates.

Run:  python payload_range.py   ->  prints the table and saves payload_range.png
"""

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- Design state from the OPTIMIZER's final output ------------------------ #
# MTOW and installed battery mass come from the optimizer's final design
# (final_design/results/data/characteristics.json); the mission propulsion
# energy is recomputed at that MTOW via energy.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from class_II_sizing.energy import cruise_power, mission_energy
from parameters import (
    V_CRUISE, M_PAYLOAD, E_AUX_KWH, RESERVE_FRAC, T_CRUISE,
    E_CELL_WH, M_CELL_KG, CELL_TO_PACK,
)

_cs = json.loads((PROJECT_ROOT / "final_design/results/data/characteristics.json").read_text())
MTOW_DESIGN   = _cs["mass"]["mtow"]                  # [kg]  optimizer final MTOW
_batt_kg      = _cs["mass"]["battery"]               # [kg]  optimizer installed battery mass
E_INSTALLED   = (_batt_kg * CELL_TO_PACK / M_CELL_KG) * E_CELL_WH / 1000.0  # [kWh] installed pack energy
E_MISSION_KWH = mission_energy(MTOW_DESIGN) / 3.6e6  # [kWh] propulsion mission energy at design MTOW
DESIGN_RANGE_KM = 200.0   # [km]  design mission range (anchor)
PAX_MASS_KG   = 100.0     # [kg]  mass per passenger incl. baggage (400 kg / 4)
N_PAX_MAX     = 4
# RESERVE_FRAC, E_AUX_KWH, T_CRUISE now imported from parameters.py above.

MASS_NO_PAYLOAD = MTOW_DESIGN - M_PAYLOAD   # everything except payload


def _design_overhead_J():
    """Non-cruise (vertical + climb + descent) energy at design MTOW [J]."""
    e_cruise_design_J = cruise_power(MTOW_DESIGN) * T_CRUISE
    return E_MISSION_KWH * 3.6e6 - e_cruise_design_J


def cruise_range_km(mtow_kg):
    """
    Cruise range achievable at a given MTOW on the fixed pack.
    Usable energy (after reserve) minus mass-scaled overhead and aux, divided
    by cruise power at this mass, times cruise speed.
    """
    e_usable_J = (E_INSTALLED / (1.0 + RESERVE_FRAC)) * 3.6e6   # to reserve point
    overhead_J = _design_overhead_J() * (mtow_kg / MTOW_DESIGN) # scales with mass
    e_aux_J = E_AUX_KWH * 3.6e6
    e_cruise_avail_J = e_usable_J - overhead_J - e_aux_J
    t_cruise_s = e_cruise_avail_J / cruise_power(mtow_kg)
    return V_CRUISE * t_cruise_s / 1000.0


def payload_range_curve():
    """Range vs payload, anchored so 4 pax == DESIGN_RANGE_KM."""
    r_design = cruise_range_km(MTOW_DESIGN)          # model range at 4 pax
    anchor = DESIGN_RANGE_KM - r_design              # shift to hit 200 km
    rows = []
    for pax in range(N_PAX_MAX, -1, -1):
        payload = pax * PAX_MASS_KG
        mtow = MASS_NO_PAYLOAD + payload
        rng = cruise_range_km(mtow) + anchor
        rows.append((pax, payload, mtow, rng))
    return rows


def main():
    rows = payload_range_curve()
    print(f"{'pax':>4}{'payload[kg]':>12}{'MTOW[kg]':>10}{'range[km]':>11}")
    for pax, payload, mtow, rng in rows:
        print(f"{pax:>4}{payload:>12.0f}{mtow:>10.0f}{rng:>11.1f}")

    # slope (km gained per passenger removed)
    r_full = rows[0][3]
    r_empty = rows[-1][3]
    print(f"\nRange at 4 pax : {r_full:.0f} km (design)")
    print(f"Range at 0 pax : {r_empty:.0f} km")
    print(f"Average gain   : {(r_empty - r_full) / N_PAX_MAX:.1f} km per passenger removed")

    payloads = [r[1] for r in rows]
    ranges = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(payloads, ranges, "-o", color="#185FA5", linewidth=2, markersize=6)
    # mark the design point
    ax.plot([M_PAYLOAD], [DESIGN_RANGE_KM], "o", color="#D85A30",
            markersize=10, zorder=5)
    ax.set_xlabel("payload [kg]  (≈100 kg per passenger)")
    ax.set_ylabel("range [km]")
    ax.set_title("Payload-range, fixed-battery eVTOL (slopes up: shed payload → more range)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 440)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()