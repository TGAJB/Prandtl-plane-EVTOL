"""
file to check the influence of the cl_max and taper ratio 
help for the airfoil and taper ratio choice
can be use for the validation of the airfoil/taper ratio choice
"""

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import (
    G,
    RHO_ORIGIN,
    V_CRUISE,
    hcruise,
    NUMBER_OF_WINGS,
    AREA_SPLIT,
    WING_SPAN,
)


class WingSizing:
    def __init__(
        self,
        mtow_kg=2000.0,
        wing_loading_n_m2=760.0,
        cruise_altitude_m=hcruise,
        cruise_speed_m_s=V_CRUISE,
        stall_density_kg_m3=RHO_ORIGIN,
        number_of_wings=NUMBER_OF_WINGS,
        area_split=AREA_SPLIT,
        wing_span_m=WING_SPAN,
        cl_max=2.0,
        taper_ratio=0.8,
    ):
        self.mtow_kg = mtow_kg
        self.wing_loading_n_m2 = wing_loading_n_m2
        self.cruise_altitude_m = cruise_altitude_m
        self.cruise_speed_m_s = cruise_speed_m_s
        self.stall_density_kg_m3 = stall_density_kg_m3
        self.number_of_wings = number_of_wings
        self.area_split = area_split
        self.wing_span_m = wing_span_m
        self.cl_max = cl_max
        self.taper_ratio = taper_ratio
        self.results = None

    @staticmethod
    def compute_isa_density(altitude_m):
        temperature_k = 288.15 - 0.0065 * altitude_m
        pressure_pa = 101325.0 * (temperature_k / 288.15) ** 5.25588
        return pressure_pa / (287.05 * temperature_k)

    def compute(self):
        weight_n = self.mtow_kg * G
        total_area_m2 = weight_n / self.wing_loading_n_m2
        area_per_wing_m2 = total_area_m2 * self.area_split

        span_m = self.wing_span_m
        aspect_ratio = span_m ** 2 / total_area_m2

        average_chord_total_m = total_area_m2 / span_m
        average_chord_per_wing_m = area_per_wing_m2 / span_m

        root_chord_m = (2.0 * area_per_wing_m2) / (span_m * (1.0 + self.taper_ratio))
        tip_chord_m = self.taper_ratio * root_chord_m

        cruise_density_kg_m3 = self.compute_isa_density(self.cruise_altitude_m)
        q_cruise = 0.5 * cruise_density_kg_m3 * self.cruise_speed_m_s ** 2
        cruise_lift_coefficient = weight_n / (q_cruise * total_area_m2)

        stall_speed_m_s = math.sqrt(
            (2.0 * weight_n) / (self.stall_density_kg_m3 * total_area_m2 * self.cl_max)
        )

        self.results = {
            "weight_n": weight_n,
            "total_area_m2": total_area_m2,
            "area_per_wing_m2": area_per_wing_m2,
            "aspect_ratio": aspect_ratio,
            "span_m": span_m,
            "average_chord_total_m": average_chord_total_m,
            "average_chord_per_wing_m": average_chord_per_wing_m,
            "root_chord_m": root_chord_m,
            "tip_chord_m": tip_chord_m,
            "cruise_density_kg_m3": cruise_density_kg_m3,
            "cruise_lift_coefficient": cruise_lift_coefficient,
            "stall_speed_m_s": stall_speed_m_s,
        }
        return self.results

    def print_summary(self):
        if self.results is None:
            self.compute()

        print(f"Aircraft weight W = {self.results['weight_n']:.2f} N")
        print(f"Total reference wing area S = {self.results['total_area_m2']:.2f} m^2")
        print(f"Number of wings = {self.number_of_wings}")
        print(f"Area per wing = {self.results['area_per_wing_m2']:.2f} m^2")
        print(f"Aspect ratio AR = {self.results['aspect_ratio']:.2f}")
        print(f"Span b = {self.results['span_m']:.2f} m")
        print(f"Average chord based on total area = {self.results['average_chord_total_m']:.2f} m")
        print(f"Average chord per wing = {self.results['average_chord_per_wing_m']:.2f} m")
        print(f"Root chord = {self.results['root_chord_m']:.2f} m")
        print(f"Tip chord = {self.results['tip_chord_m']:.2f} m")
        print(f"Cruise density rho_cruise = {self.results['cruise_density_kg_m3']:.3f} kg/m^3")
        print(f"Cruise lift coefficient CL_cruise = {self.results['cruise_lift_coefficient']:.3f}")
        print(f"Stall speed V_stall = {self.results['stall_speed_m_s']:.2f} m/s")


def main():
    wing_sizing = WingSizing()
    wing_sizing.compute()
    wing_sizing.print_summary()


if __name__ == "__main__":
    main()
