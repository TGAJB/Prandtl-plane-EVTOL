"""
battery_cells.py -- Battery pack cell-count sizing for the Folding Prandtl eVTOL.

BASELINE CELL: Amprius SA504 SiCore (High Energy + High Power) silicon-anode
Li-ion, a real, commercially-available pouch cell. Datasheet: 386 Wh/kg
cell-level, 6C continuous discharge, 500 cycles (100% DOD to 80% SOH). Chosen
over the SA88 (365 Wh/kg, 10C, 200 cycles) because its higher energy density
and 2.5x longer cycle life outweigh the lower (but still adequate) power
rating: hover needs ~4.4C at pack level, which the 6C cell covers with a
~1.35x margin. Lower-power higher-energy SiCore variants (1C) cannot meet hover.

PARAMETER SET:
  cell-level specific energy : 386 Wh/kg  (SA504, datasheet)
  cell-to-pack ratio         : 0.72       (realistic integrated pack)
  energy margin              : 20% reserve (no separate SOC/contingency)
  mass allocation            : 546 kg     (Class-II estimate, with margin)
"""

"""
battery_cells.py -- Battery pack cell-count sizing for the Folding Prandtl eVTOL.

BASELINE CELL: Amprius SA504 SiCore (High Energy + High Power) silicon-anode
Li-ion, a real, commercially-available pouch cell. Datasheet: 386 Wh/kg
cell-level, 6C continuous discharge, 500 cycles (100% DOD to 80% SOH). Chosen
over the SA88 (365 Wh/kg, 10C, 200 cycles) because its higher energy density
and 2.5x longer cycle life outweigh the lower (but still adequate) power
rating: hover needs ~4.4C at pack level, which the 6C cell covers with a
~1.35x margin. Lower-power higher-energy SiCore variants (1C) cannot meet hover.

PARAMETER SET:
  cell-level specific energy : 386 Wh/kg  (SA504, datasheet)
  cell-to-pack ratio         : 0.72       (realistic integrated pack)
  energy margin              : 20% reserve (no separate SOC/contingency)
  mass allocation            : 546 kg     (Class-II estimate, with margin)
"""

from dataclasses import dataclass
import math


import math


# --------------------------------------------------------------------------- #
# Cell datasheet
# --------------------------------------------------------------------------- #
@dataclass
class Cell:
    name: str = "Amprius SA504 SiCore (High Energy + High Power)"
    V_nom: float = 3.40         # nominal voltage [V] (datasheet)
    capacity_Ah: float = 11.05  # typical capacity @ C/5 [Ah] (datasheet)
    # SA504: 37.57 Wh typical, 97.3 g -> 386 Wh/kg; 66.3 A (6C) continuous.
    I_cont_max: float = 66.3    # max continuous discharge current [A] (6C)
    mass_kg: float = 0.0973     # cell mass [kg] (datasheet, 97.3 g)
    assumed_density_Wh_kg: float = 386.0  # cell-level density (datasheet)
    cycle_life: int = 500       # cycles, 1C/-1C, 100% DOD to 80% SOH (datasheet)
    clamp_psi: float = 0.0      # pouch; clamping not specified on datasheet

    name: str = "Amprius SA504 SiCore (High Energy + High Power)"
    V_nom: float = 3.40         # nominal voltage [V] (datasheet)
    capacity_Ah: float = 11.05  # typical capacity @ C/5 [Ah] (datasheet)
    # SA504: 37.57 Wh typical, 97.3 g -> 386 Wh/kg; 66.3 A (6C) continuous.
    I_cont_max: float = 66.3    # max continuous discharge current [A] (6C)
    mass_kg: float = 0.0973     # cell mass [kg] (datasheet, 97.3 g)
    assumed_density_Wh_kg: float = 386.0  # cell-level density (datasheet)
    cycle_life: int = 500       # cycles, 1C/-1C, 100% DOD to 80% SOH (datasheet)
    clamp_psi: float = 0.0      # pouch; clamping not specified on datasheet

    @property
    def energy_Wh(self) -> float:
        """Per-cell energy at nominal voltage [Wh] (datasheet typical 37.57)."""
        return self.V_nom * self.capacity_Ah     # 3.40 * 11.05 = 37.57 Wh

        """Per-cell energy at nominal voltage [Wh] (datasheet typical 37.57)."""
        return self.V_nom * self.capacity_Ah     # 3.40 * 11.05 = 37.57 Wh

    @property
    def energy_Wh_assumed(self) -> float:
        """Energy-driven count uses the real datasheet per-cell energy."""
        return self.energy_Wh


    @property
    def power_cont_W(self) -> float:
        """Continuous power per cell at nominal voltage [W] (6C)."""
        return self.V_nom * self.I_cont_max      # 3.40 * 66.3 = 225 W

        """Continuous power per cell at nominal voltage [W] (6C)."""
        return self.V_nom * self.I_cont_max      # 3.40 * 66.3 = 225 W

    @property
    def density_Wh_kg(self) -> float:
        return self.energy_Wh / self.mass_kg     # ~386 Wh/kg
        return self.energy_Wh / self.mass_kg     # ~386 Wh/kg


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Sizing inputs
# --------------------------------------------------------------------------- #
@dataclass
class PackRequirement:
    # ENERGY side
    # 137.2 kWh = mission_energy(2314.08)/3.6e6 from the converged MTOW loop
    # (mtow_sizing.py, SA504 battery_mass in the loop). Update if MTOW moves.
    E_mission_kWh: float = 137.2     # main PROPULSION mission energy [kWh]
    E_aux_kWh: float = 0.0           # auxiliary (270V+28V) energy [kWh];
                                     # set from aux_loads.py in main
                                     # set from aux_loads.py in main
    E_reserve_frac: float = 0.20     # reserve = 20% of (mission + aux)
    usable_soc: float = 1.00         # no separate SOC derate; reserve is margin
    contingency: float = 1.00        # no separate contingency (reserve is it)


    # POWER side
    # 870 kW = takeoff_power(2314.08) from energy.py at the converged MTOW
    # (the mission CSV's 667 kW was computed at MTOW 2000 and is stale;
    # regenerate the CSV at 2314 for report traceability). Power is not the
    # binding constraint either way -- energy governs.
    P_hover_kW: float = 870.0        # hover/takeoff power [kW] (energy.py @ 2314)
    P_aux_concurrent_kW: float = 0.0 # aux power concurrent with hover [kW];
                                     # realistic basis -- propulsion dominates
    power_derate: float = 1.0        # optional pack-level derate on cell I_cont


    @property
    def E_reserve_kWh(self) -> float:
        return (self.E_mission_kWh + self.E_aux_kWh) * self.E_reserve_frac


    # Architecture target (for reporting S/P split)
    bus_voltage_V: float = 800.0     # high-voltage bus target [V]

    # Mass reference -- the converged MTOW loop's own battery line (the old
    # 546 kg Class-II allocation is superseded; the loop now sizes the battery
    # with this same method, so the "budget" IS the converged output).
    pack_mass_budget_kg: float = 635.15  # converged loop battery mass @ MTOW 2314
    cell_to_pack: float = 0.72           # realistic integrated pack


# --------------------------------------------------------------------------- #
# Cell-count sizing
# --------------------------------------------------------------------------- #
def cells_for_energy(cell: Cell, req: PackRequirement) -> dict:
    """
    Energy-driven cell count.
        E_deliverable = E_mission + E_aux + E_reserve
        E_installed   = E_deliverable * contingency / usable_soc
    Energy-driven cell count.
        E_deliverable = E_mission + E_aux + E_reserve
        E_installed   = E_deliverable * contingency / usable_soc
        N = ceil(E_installed / E_cell)
    """
    E_deliverable_kWh = req.E_mission_kWh + req.E_aux_kWh + req.E_reserve_kWh
    E_installed_kWh = E_deliverable_kWh * req.contingency / req.usable_soc
    n = E_installed_kWh * 1000.0 / cell.energy_Wh_assumed
    N = math.ceil(n)
    return {
        "E_deliverable_kWh": E_deliverable_kWh,
        "E_installed_kWh": E_installed_kWh,
        "N_cells_exact": n,
        "N_cells": N,
        "pack_mass_cells_kg": N * cell.mass_kg,
    }




def cells_for_power(cell: Cell, req: PackRequirement) -> dict:
    """
    Power-driven cell count.
    Power-driven cell count.
        P_cell = V_nom * I_cont_max * power_derate
        N = ceil(P_hover / P_cell)
    """
    P_cell_W = cell.power_cont_W * req.power_derate
    P_demand_kW = req.P_hover_kW + req.P_aux_concurrent_kW
    n = P_demand_kW * 1000.0 / P_cell_W
    N = math.ceil(n)
    return {
        "P_cell_W": P_cell_W,
        "P_demand_kW": P_demand_kW,
        "N_cells_exact": n,
        "N_cells": N,
        "pack_mass_cells_kg": N * cell.mass_kg,
    }




def series_parallel(cell: Cell, N_cells: int, req: PackRequirement) -> dict:
    """
    Series/parallel layout for the chosen cell count and bus voltage.
        S = round(bus_voltage / V_nom)   (series sets pack voltage)
        P = ceil(N_cells / S)            (parallel strings for energy/power)
    Series/parallel layout for the chosen cell count and bus voltage.
        S = round(bus_voltage / V_nom)   (series sets pack voltage)
        P = ceil(N_cells / S)            (parallel strings for energy/power)
    """
    S = round(req.bus_voltage_V / cell.V_nom)
    P = math.ceil(N_cells / S)
    N_actual = S * P
    return {
        "S": S, "P": P, "N_actual": N_actual,
        "pack_voltage_V": S * cell.V_nom,
        "pack_energy_kWh": N_actual * cell.energy_Wh_assumed / 1000.0,
        "pack_power_cont_kW": N_actual * cell.power_cont_W / 1000.0,
        "pack_mass_cells_kg": N_actual * cell.mass_kg,
    }




def cells_from_mass_budget(cell: Cell, req: PackRequirement) -> dict:
    """
    How many cells fit in the allocated pack mass, and what they deliver.
        cell mass available = pack_mass_budget * cell_to_pack
        N_fit = floor(cell_mass_available / cell_mass)
    """
    cell_mass_available_kg = req.pack_mass_budget_kg * req.cell_to_pack
    N_fit = math.floor(cell_mass_available_kg / cell.mass_kg)
    E_installed_kWh = N_fit * cell.energy_Wh_assumed / 1000.0
    E_deliverable_kWh = E_installed_kWh * req.usable_soc
    E_required_kWh = (req.E_mission_kWh + req.E_aux_kWh + req.E_reserve_kWh) * req.contingency
    P_avail_kW = N_fit * cell.power_cont_W / 1000.0
    return {
        "cell_mass_available_kg": cell_mass_available_kg,
        "N_fit": N_fit,
        "E_installed_kWh": E_installed_kWh,
        "E_deliverable_kWh": E_deliverable_kWh,
        "E_required_kWh": E_required_kWh,
        "energy_margin_kWh": E_deliverable_kWh - E_required_kWh,
        "energy_feasible": E_deliverable_kWh >= E_required_kWh,
        "P_avail_kW": P_avail_kW,
        "power_feasible": P_avail_kW >= req.P_hover_kW,
    }




def size_pack(cell: Cell, req: PackRequirement) -> dict:
    e = cells_for_energy(cell, req)
    p = cells_for_power(cell, req)
    driver = "ENERGY" if e["N_cells"] >= p["N_cells"] else "POWER"
    N = max(e["N_cells"], p["N_cells"])
    layout = series_parallel(cell, N, req)
    return {"energy": e, "power": p, "driver": driver,
            "N_required": N, "layout": layout}




# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report(cell: Cell, req: PackRequirement) -> None:
    r = size_pack(cell, req)
    print("=" * 70)
    print(f"BATTERY CELL-COUNT SIZING -- {cell.name}")
    print("=" * 70)
    print(f"Cell: {cell.V_nom} V x {cell.capacity_Ah} Ah = {cell.energy_Wh:.1f} Wh "
          f"({cell.assumed_density_Wh_kg:.0f} Wh/kg), "
          f"{cell.I_cont_max:.0f} A -> {cell.power_cont_W:.0f} W, {cell.mass_kg*1000:.0f} g")
    print("-" * 70)
    print("ENERGY constraint (per-cell energy at the SA504 datasheet density):")
    print(f"  ({req.E_mission_kWh:.1f} mission + {req.E_aux_kWh:.2f} aux + "
          f"{req.E_reserve_kWh:.1f} reserve) "
          f"= {r['energy']['E_deliverable_kWh']:.1f} kWh deliverable")
    print(f"  x {req.contingency} contingency / {req.usable_soc} usable "
          f"= {r['energy']['E_installed_kWh']:.1f} kWh installed")
    print(f"  -> {r['energy']['N_cells']} cells "
          f"({r['energy']['pack_mass_cells_kg']:.0f} kg of cells)")
    print("POWER constraint:")
    print(f"  hover {req.P_hover_kW:.0f} kW / ({cell.power_cont_W:.0f} W x "
          f"{req.power_derate} derate) per cell")
    print(f"  -> {r['power']['N_cells']} cells "
          f"({r['power']['pack_mass_cells_kg']:.0f} kg of cells)")
    print("-" * 70)
    print(f"DRIVING CONSTRAINT: {r['driver']}  ->  {r['N_required']} cells required")
    L = r["layout"]
    print(f"\nSuggested layout for {req.bus_voltage_V:.0f} V bus:")
    print(f"  {L['S']}S x {L['P']}P = {L['N_actual']} cells")
    print(f"  pack voltage : {L['pack_voltage_V']:.0f} V")
    print(f"  pack energy  : {L['pack_energy_kWh']:.1f} kWh")
    print(f"  pack power   : {L['pack_power_cont_kW']:.0f} kW continuous")
    print(f"  cell mass    : {L['pack_mass_cells_kg']:.0f} kg "
          f"(/ {req.cell_to_pack} cell-to-pack -> "
          f"{L['pack_mass_cells_kg']/req.cell_to_pack:.0f} kg pack)")
    print("=" * 70)

    # --- Mass-allocation feasibility check ----------------------------- #
    m = cells_from_mass_budget(cell, req)
    print(f"\nMASS-ALLOCATION FEASIBILITY (Class-II pack mass "
          f"{req.pack_mass_budget_kg:.0f} kg, cell-to-pack {req.cell_to_pack})")
    print(f"  cell mass available : {m['cell_mass_available_kg']:.0f} kg "
          f"-> fits {m['N_fit']} cells")
    print(f"  energy installed    : {m['E_installed_kWh']:.1f} kWh")
    print(f"  energy deliverable  : {m['E_deliverable_kWh']:.1f} kWh "
          f"(usable SOC {req.usable_soc})")
    print(f"  energy required     : {m['E_required_kWh']:.1f} kWh "
          f"(mission+aux+reserve x contingency)")
    print(f"  energy margin       : {m['energy_margin_kWh']:+.1f} kWh  "
          f"[{'FEASIBLE' if m['energy_feasible'] else 'SHORTFALL'}]")
    print(f"  power available     : {m['P_avail_kW']:.0f} kW vs "
          f"{req.P_hover_kW:.0f} kW hover  "
          f"[{'FEASIBLE' if m['power_feasible'] else 'SHORTFALL'}]")
    print("=" * 70)




if __name__ == "__main__":
    cell = Cell()


    # --- Auxiliary energy from the component build-up (aux_loads.py) ----- #
    # Single source of truth: edit loads/duties in aux_loads.py, not here.
    from aux_loads import default_aux_loads as _full_aux, energy_breakdown
    _aux_b = energy_breakdown(_full_aux())
    aux_total = _aux_b["total_kWh"]
    print("=" * 70)
    print("AUXILIARY ENERGY (from component build-up, aux_loads.py)")
    print("=" * 70)
    for p in ["Takeoff", "Vertical climb", "Climb", "Cruise", "Descent", "Landing"]:
        print(f"  {p:<16}{_aux_b['phase_E'][p]:>8.3f} kWh")
    print(f"  {'28 V bus':<16}{_aux_b['bus_E'][28]:>8.2f} kWh")
    print(f"  {'270 V bus':<16}{_aux_b['bus_E'][270]:>8.2f} kWh")
    print(f"  {'TOTAL':<16}{aux_total:>8.2f} kWh")
    print("=" * 70 + "\n")

    # Energy governs; propulsion dominates the hover peak so concurrent aux ~ 0.
    req = PackRequirement(E_aux_kWh=aux_total, P_aux_concurrent_kW=0.0)
    report(cell, req)