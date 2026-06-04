from dataclasses import dataclass
 
 
# Phase durations [s] from the team constants file
PHASES = [
    ("Takeoff",         5.0),
    ("Vertical climb",  25.0),
    ("Climb",           1221.7),
    ("Cruise",          2194.7),
    ("Descent",         311.945),
    ("Landing",         71.47),
]
 
ETA_DCDC = 0.95   # DC/DC conversion efficiency (battery delivers more than bus uses)
 
 
@dataclass
class AuxLoad:
    name: str
    bus_V: int                 # 28 or 270
    watts: float               # nominal continuous draw when active [W]
    duty: dict                 # phase name -> fraction active [0..1]
    tag: str = "[PH]"          # provenance
    owner: str = ""
    note: str = ""
 
    def active_W(self, phase: str) -> float:
        return self.watts * self.duty.get(phase, 0.0)
 
 
# Duty-cycle shorthand helpers
ALWAYS = {p: 1.0 for p, _ in PHASES}                       # on the whole flight
VERT   = {"Takeoff": 1.0, "Vertical climb": 1.0, "Climb": 0.2,
          "Cruise": 0.0, "Descent": 0.2, "Landing": 1.0}    # actuator-heavy phases
GROUND = {p: 0.0 for p, _ in PHASES}                        # not in flight
 
 
def default_aux_loads() -> list:
    """
    Full auxiliary load list. Avionics rows tagged [ECS] are carried from the
    ECS-model build-up (same numbers). All others are placeholders [PH] with the
    owner who can supply the real value. Edit watts and duty cycles freely.
    """
    return [
        # --- 28 V bus: flight-critical avionics & sensors (from ECS build-up) ---
        AuxLoad("Flight computers (x3)",     28, 150.0, ALWAYS, "[ECS]", "Control", "3 x 50 W redundant"),
        AuxLoad("IMU (x2)",                  28, 4.0,   ALWAYS, "[ECS]", "Control", "2 x 2 W"),
        AuxLoad("GNSS (x2)",                 28, 6.0,   ALWAYS, "[ECS]", "Control", "2 x 3 W"),
        AuxLoad("Air-data sensors",          28, 5.0,   ALWAYS, "[ECS]", "Control", "pitot/AoA/baro"),
        AuxLoad("ADS-B transponder",         28, 3.0,   ALWAYS, "[ECS]", "Control", "uAvionix ping200X ~1.5-3 W"),
        AuxLoad("LiDAR",                     28, 90.0,  ALWAYS, "[src]", "Control", "scanning aviation LiDAR, datasheet"),
        AuxLoad("Radar",                     28, 30.0,  ALWAYS, "[ECS]", "Control", "vehicle avoidance"),
        AuxLoad("SATCOM terminal",           28, 150.0, ALWAYS, "[ECS]", "Control", "LEO link; dominant avionics load"),
        AuxLoad("Cellular modem",            28, 15.0,  ALWAYS, "[ECS]", "Control", "backup link"),
        AuxLoad("Cameras (5)",               28, 20.0,  ALWAYS, "[ECS]", "Control", "4 ext + 1 int, ~4 W each"),
 
        # --- 28 V bus: cabin & misc (placeholders) ---
        AuxLoad("Cabin + nav lighting",      28, 80.0,  ALWAYS, "[PH]",  "Cabin",   "LED interior + nav/anti-collision"),
        AuxLoad("Displays + infotainment",   28, 66.0,  {"Takeoff":1,"Vertical climb":1,"Climb":1,
                                                          "Cruise":1,"Descent":1,"Landing":1}, "[src]", "Cabin",
                "4 pax screens x 9 W (=36 W; 9-in LCD per Fioriti, matches ECS "
                "display term) + ~30 W head unit/audio amp"),
        AuxLoad("USB / passenger charging",  28, 100.0, ALWAYS, "[PH]",  "Cabin",   "4 pax, assume ~25 W each peak"),
        AuxLoad("BMS logic",                 28, 30.0,  ALWAYS, "[PH]",  "Power",   "pack monitoring/contactors logic"),
        AuxLoad("Pressure / cabin sensors",  28, 10.0,  ALWAYS, "[PH]",  "Cabin",   ""),
 
        # --- 270 V bus: actuators, folding, AC (placeholders) ---
        AuxLoad("Tilt actuators",            270, 6000.0, VERT,   "[PH]", "Propulsion",
                "BIG peak, SHORT duration -> small energy. Confirm power while tilting + tilt time"),
        AuxLoad("Folding mechanism",         270, 4000.0, GROUND, "[PH]", "Mechanisms",
                "ground only -> 0 in flight; confirm it is not used airborne"),
        AuxLoad("Control-surface actuators", 270, 800.0,  ALWAYS, "[PH]", "Control",
                "EMA flight controls, low continuous"),
        AuxLoad("AC compressor",             270, 1100.0, {"Takeoff":1.0,"Vertical climb":1.0,
                                                            "Climb":0.5,"Cruise":0.0,
                                                            "Descent":0.3,"Landing":1.0},
                "[ECS-derived]", "ECS/Cabin",
                "1.1 kW = ~2.2 kW cabin cooling load (ECS model, hot case) / COP 2. "
                "Off in cruise (cabin is net-heating at altitude). SEE double-count "
                "note: if SAME machine as ECS pressurization compressor, delete this "
                "row and use the ECS model compressor energy instead."),
    ]
 
 
def energy_breakdown(loads: list) -> dict:
    """Per-phase and total auxiliary energy [kWh], including DC/DC loss."""
    phase_E = {p: 0.0 for p, _ in PHASES}
    bus_E = {28: 0.0, 270: 0.0}
    for p, dt in PHASES:
        for L in loads:
            e = L.active_W(p) / 1000.0 / ETA_DCDC * (dt / 3600.0)  # kWh
            phase_E[p] += e
            bus_E[L.bus_V] += e
    total = sum(phase_E.values())
    return {"phase_E": phase_E, "bus_E": bus_E, "total_kWh": total}
 
 
def print_table(loads: list) -> None:
    print("=" * 92)
    print("AUXILIARY ELECTRICAL LOAD BUILD-UP  ([ECS]=from ECS model, [PH]=placeholder)")
    print("=" * 92)
    print(f"{'load':<26}{'bus':>5}{'W':>8}{'tag':>7}{'owner':>12}   duty (T/VC/Cl/Cr/D/L)")
    for L in loads:
        duty = "/".join(f"{L.duty.get(p,0):.1f}" for p, _ in PHASES)
        print(f"{L.name:<26}{L.bus_V:>5}{L.watts:>8.0f}{L.tag:>7}{L.owner:>12}   {duty}")
    print("-" * 92)
    b = energy_breakdown(loads)
    print("Energy per phase [kWh]:")
    for p, _ in PHASES:
        print(f"   {p:<16}{b['phase_E'][p]:.3f}")
    print(f"By bus:  28 V = {b['bus_E'][28]:.2f} kWh   270 V = {b['bus_E'][270]:.2f} kWh")
    print(f"TOTAL auxiliary energy = {b['total_kWh']:.2f} kWh "
          f"(incl. DC/DC {ETA_DCDC})")
    print("=" * 92)
    # sanity-check anchor
    print("Sanity check: steady 28 V draw (ALWAYS-on loads) =",
          f"{sum(L.watts for L in loads if L.bus_V==28 and L.duty.get('Cruise',0)>0):.0f} W,",
          "a few kW class -- consistent with small-aircraft avionics+cabin loads.")
 
 
if __name__ == "__main__":
    loads = default_aux_loads()
    print_table(loads)