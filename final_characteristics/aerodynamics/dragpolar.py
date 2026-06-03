"""
This file outputs:
- maximum L/D and the corresponding C_L and C_D
- cruise L/D and the corresponding C_L and C_D
- maximum speed V_max from the P_a = P_r intersection
- stall speed V_stall and operational lift-coefficient limits
- a power-available / power-required plot
- a drag polar plot
- an L/D vs C_L plot
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from class_II_sizing.mtow_sizing import load_final_design_state
from parameters import (
    AR_W,
    CD0,
    CL_MAX_OPERATIONAL,
    CL_PLOT_MAX,
    CL_PLOT_MIN,
    DRAG_POLAR_N_POINTS,
    G,
    OSWALD_EFFICIENCY,
    RHO_ORIGIN,
    S_W,
    V_CRUISE,
    hcruise,
)


class PowerCurveAnalysis:
    def __init__(
        self,
        velocity_m_s=None,
        power_required_w=None,
        power_available_w=None,
    ):
        default_velocity = np.array(
            [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
            dtype=float,
        )
        default_power_required = np.array(
            [
                2.21e01,
                1.77e02,
                5.97e02,
                1.42e03,
                2.77e03,
                4.78e03,
                7.59e03,
                1.13e04,
                1.61e04,
                2.21e04,
                2.94e04,
                3.82e04,
                4.86e04,
                6.07e04,
                7.47e04,
                9.06e04,
                1.09e05,
            ],
            dtype=float,
        )

        self.velocity_m_s = np.array(
            default_velocity if velocity_m_s is None else velocity_m_s,
            dtype=float,
        )
        self.power_required_w = np.array(
            default_power_required if power_required_w is None else power_required_w,
            dtype=float,
        )

        if power_available_w is None:
            self.power_available_w = np.full_like(self.velocity_m_s, 6.05e04, dtype=float)
        elif np.isscalar(power_available_w):
            self.power_available_w = np.full_like(
                self.velocity_m_s,
                float(power_available_w),
                dtype=float,
            )
        else:
            self.power_available_w = np.array(power_available_w, dtype=float)

        self._validate_inputs()

    def _validate_inputs(self):
        if len(self.velocity_m_s) < 2:
            raise ValueError("At least two velocity points are required.")

        if not (
            len(self.velocity_m_s)
            == len(self.power_required_w)
            == len(self.power_available_w)
        ):
            raise ValueError("Velocity, power required, and power available arrays must match.")

        if not np.all(np.diff(self.velocity_m_s) > 0):
            raise ValueError("velocity_m_s must be strictly increasing.")

    def find_intersection_velocities(self):
        difference = self.power_available_w - self.power_required_w
        intersection_velocities = []

        for index in range(len(self.velocity_m_s) - 1):
            if np.isclose(difference[index], 0.0):
                intersection_velocities.append(self.velocity_m_s[index])
            elif difference[index] * difference[index + 1] < 0:
                velocity_intersection = self.velocity_m_s[index] - difference[index] * (
                    self.velocity_m_s[index + 1] - self.velocity_m_s[index]
                ) / (difference[index + 1] - difference[index])
                intersection_velocities.append(velocity_intersection)

        if np.isclose(difference[-1], 0.0):
            intersection_velocities.append(self.velocity_m_s[-1])

        return intersection_velocities

    def get_max_speed(self):
        intersection_velocities = self.find_intersection_velocities()
        if not intersection_velocities:
            raise ValueError("No P_a = P_r intersection found in the given velocity range.")
        return max(intersection_velocities)

    def plot(self, ax=None):
        intersection_velocities = self.find_intersection_velocities()
        created_figure = ax is None

        if created_figure:
            figure, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            self.velocity_m_s,
            self.power_available_w,
            marker="o",
            label=r"$P_a$ - Power available",
        )
        ax.plot(
            self.velocity_m_s,
            self.power_required_w,
            marker="s",
            label=r"$P_r$ - Power required",
        )

        for index, velocity_intersection in enumerate(intersection_velocities):
            power_intersection = np.interp(
                velocity_intersection,
                self.velocity_m_s,
                self.power_available_w,
            )
            label = None
            if index == 0:
                label = fr"$P_a=P_r$ at {velocity_intersection:.2f} m/s"
            ax.scatter(
                velocity_intersection,
                power_intersection,
                marker="x",
                s=100,
                color="tab:red",
                label=label,
                zorder=3,
            )

        ax.set_xlabel(r"Velocity $V$ [m/s]")
        ax.set_ylabel(r"Power $P$ [W]")
        ax.set_title("Power Available and Power Required vs Velocity")
        ax.grid(True)
        ax.legend()

        if created_figure:
            figure.tight_layout()

        return ax


class DragPolarAnalysis:
    def __init__(
        self,
        mass_kg=None,
        gravity_m_s2=G,
        cruise_altitude_m=hcruise,
        stall_density_kg_m3=RHO_ORIGIN,
        wing_area_m2=None,
        cruise_speed_m_s=V_CRUISE,
        cl_max_operational=CL_MAX_OPERATIONAL,
        cl_min=CL_PLOT_MIN,
        cl_max=CL_PLOT_MAX,
        cd0=CD0,
        aspect_ratio=None,
        oswald_efficiency=OSWALD_EFFICIENCY,
        n_points=DRAG_POLAR_N_POINTS,
        velocity_m_s=None,
        power_required_w=None,
        power_available_w=None,
    ):
        design_state = None
        wing_sizing_final = None

        if mass_kg is None or wing_area_m2 is None or aspect_ratio is None:
            design_state = load_final_design_state(verbose=False)
            wing_sizing_final = design_state["wing_sizing_final"]

        default_mass_kg = (
            design_state["mtow_final_kg"] if design_state is not None else None
        )
        default_wing_area_m2 = (
            wing_sizing_final["total_area_m2"] if wing_sizing_final is not None else S_W
        )
        default_aspect_ratio = (
            wing_sizing_final["aspect_ratio"] if wing_sizing_final is not None else AR_W
        )

        self.mass_kg = default_mass_kg if mass_kg is None else mass_kg
        self.gravity_m_s2 = gravity_m_s2
        self.cruise_altitude_m = cruise_altitude_m
        self.stall_density_kg_m3 = stall_density_kg_m3
        self.wing_area_m2 = default_wing_area_m2 if wing_area_m2 is None else wing_area_m2
        self.cruise_speed_m_s = cruise_speed_m_s
        self.cl_max_operational = cl_max_operational
        self.cl_min = cl_min
        self.cl_max = cl_max
        self.cd0 = cd0
        self.aspect_ratio = default_aspect_ratio if aspect_ratio is None else aspect_ratio
        self.oswald_efficiency = oswald_efficiency
        self.n_points = n_points
        self.power_curve = PowerCurveAnalysis(
            velocity_m_s=velocity_m_s,
            power_required_w=power_required_w,
            power_available_w=power_available_w,
        )
        self.results = None

    @staticmethod
    def compute_isa_density(altitude_m):
        temperature_k = 288.15 - 0.0065 * altitude_m
        pressure_pa = 101325.0 * (temperature_k / 288.15) ** 5.25588
        return pressure_pa / (287.05 * temperature_k)

    @staticmethod
    def _add_horizontal_limit_if_visible(ax, value, lower_bound, upper_bound, label):
        if lower_bound <= value <= upper_bound:
            ax.axhline(value, color="red", linestyle="--", linewidth=1.5, label=label)

    @staticmethod
    def _add_vertical_limit_if_visible(ax, value, lower_bound, upper_bound, label):
        if lower_bound <= value <= upper_bound:
            ax.axvline(value, color="red", linestyle="--", linewidth=1.5, label=label)

    def _validate_inputs(self):
        if self.mass_kg is None:
            raise ValueError("mass_kg could not be resolved from class_II_sizing/mtow_sizing.py.")

        if self.cl_min >= self.cl_max:
            raise ValueError("cl_min must be smaller than cl_max.")

        if self.cl_max_operational <= 0:
            raise ValueError("cl_max_operational must be positive.")

        if self.stall_density_kg_m3 <= 0:
            raise ValueError("stall_density_kg_m3 must be positive.")

        if self.wing_area_m2 <= 0:
            raise ValueError("wing_area_m2 must be positive.")

        if self.aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be positive.")

        if self.oswald_efficiency <= 0:
            raise ValueError("oswald_efficiency must be positive.")

        if self.n_points < 2:
            raise ValueError("n_points must be at least 2.")

    def compute(self):
        self._validate_inputs()

        weight_n = self.mass_kg * self.gravity_m_s2
        max_speed_m_s = self.power_curve.get_max_speed()
        cruise_density_kg_m3 = self.compute_isa_density(self.cruise_altitude_m)

        stall_speed_m_s = np.sqrt(
            2.0 * weight_n
            / (self.stall_density_kg_m3 * self.wing_area_m2 * self.cl_max_operational)
        )
        cl_min_operational = (
            2.0 * weight_n
            / (cruise_density_kg_m3 * max_speed_m_s**2 * self.wing_area_m2)
        )

        if stall_speed_m_s >= max_speed_m_s:
            raise ValueError("V_stall must be smaller than V_max.")

        induced_drag_factor = 1.0 / (
            np.pi * self.aspect_ratio * self.oswald_efficiency
        )

        cl_values = np.linspace(self.cl_min, self.cl_max, self.n_points)
        cd_values = self.cd0 + induced_drag_factor * cl_values**2
        ld_values = cl_values / cd_values

        q_cruise = 0.5 * cruise_density_kg_m3 * self.cruise_speed_m_s**2
        cl_cruise = weight_n / (q_cruise * self.wing_area_m2)
        cd_cruise = self.cd0 + induced_drag_factor * cl_cruise**2
        ld_cruise = cl_cruise / cd_cruise

        max_ld_index = np.argmax(ld_values)
        cl_at_max_ld = cl_values[max_ld_index]
        cd_at_max_ld = cd_values[max_ld_index]
        max_ld = ld_values[max_ld_index]

        self.results = {
            "weight_n": weight_n,
            "cruise_density_kg_m3": cruise_density_kg_m3,
            "stall_density_kg_m3": self.stall_density_kg_m3,
            "max_speed_m_s": max_speed_m_s,
            "stall_speed_m_s": stall_speed_m_s,
            "cl_min_operational": cl_min_operational,
            "cl_max_operational": self.cl_max_operational,
            "induced_drag_factor": induced_drag_factor,
            "cl_values": cl_values,
            "cd_values": cd_values,
            "ld_values": ld_values,
            "cl_cruise": cl_cruise,
            "cd_cruise": cd_cruise,
            "ld_cruise": ld_cruise,
            "cl_at_max_ld": cl_at_max_ld,
            "cd_at_max_ld": cd_at_max_ld,
            "max_ld": max_ld,
            "power_intersection_velocities_m_s": self.power_curve.find_intersection_velocities(),
        }
        return self.results

    def print_summary(self):
        if self.results is None:
            self.compute()

        print(f"MTOW from class_II_sizing/mtow_sizing.py = {self.mass_kg:.2f} kg")
        print(f"Wing area from class_II_sizing/mtow_sizing.py = {self.wing_area_m2:.2f} m^2")
        print(f"Aspect ratio from class_II_sizing/mtow_sizing.py = {self.aspect_ratio:.3f}")
        print(f"Cruise altitude from parameters.py = {self.cruise_altitude_m:.2f} m")
        print(f"Cruise density = {self.results['cruise_density_kg_m3']:.4f} kg/m^3")
        print(f"Stall density = {self.results['stall_density_kg_m3']:.4f} kg/m^3")
        print(f"CL_min_operational = {self.results['cl_min_operational']:.3f}")
        print(f"CL_max_operational = {self.results['cl_max_operational']:.3f}")
        print(f"Using V_max from P_a = P_r = {self.results['max_speed_m_s']:.2f} m/s")
        print(f"Calculated V_stall = {self.results['stall_speed_m_s']:.2f} m/s")
        print(f"Using cruise speed V_cruise = {self.cruise_speed_m_s:.2f} m/s")
        print(f"C_L_min_input = {self.cl_min:.3f}")
        print(f"C_L_max_input = {self.cl_max:.3f}")

        if not (self.cl_min <= self.results["cl_min_operational"] <= self.cl_max):
            print("Operational C_L_min is outside the input C_L plotting range.")

        if not (self.cl_min <= self.cl_max_operational <= self.cl_max):
            print("Operational C_L_max is outside the input C_L plotting range.")

        if not (self.cl_min <= self.results["cl_cruise"] <= self.cl_max):
            print("Cruise C_L is outside the input C_L plotting range.")

        print(f"Induced drag factor k = {self.results['induced_drag_factor']:.4f}")
        print(f"Maximum L/D = {self.results['max_ld']:.2f}")
        print(f"CL at maximum L/D = {self.results['cl_at_max_ld']:.3f}")
        print(f"CD at maximum L/D = {self.results['cd_at_max_ld']:.4f}")
        print(f"Cruise L/D = {self.results['ld_cruise']:.2f}")
        print(f"CL at cruise = {self.results['cl_cruise']:.3f}")
        print(f"CD at cruise = {self.results['cd_cruise']:.4f}")

        intersection_velocities = self.results["power_intersection_velocities_m_s"]
        if intersection_velocities:
            for velocity_intersection in intersection_velocities:
                print(f"P_a = P_r at approximately V = {velocity_intersection:.2f} m/s")

    def plot_drag_polar(self, ax=None):
        if self.results is None:
            self.compute()

        created_figure = ax is None

        if created_figure:
            figure, ax = plt.subplots(figsize=(7, 5))

        ax.plot(
            self.results["cd_values"],
            self.results["cl_values"],
            label="Drag polar from input $C_L$ range",
        )
        ax.scatter(
            self.results["cd_at_max_ld"],
            self.results["cl_at_max_ld"],
            marker="o",
            color="tab:green",
            zorder=3,
            label=f"Max L/D = {self.results['max_ld']:.2f}",
        )

        if self.cl_min <= self.results["cl_cruise"] <= self.cl_max:
            ax.scatter(
                self.results["cd_cruise"],
                self.results["cl_cruise"],
                marker="s",
                color="tab:orange",
                zorder=3,
                label=f"Cruise L/D = {self.results['ld_cruise']:.2f}",
            )

        self._add_horizontal_limit_if_visible(
            ax,
            self.results["cl_min_operational"],
            self.cl_min,
            self.cl_max,
            r"Operational $C_{L,\min}$",
        )
        self._add_horizontal_limit_if_visible(
            ax,
            self.cl_max_operational,
            self.cl_min,
            self.cl_max,
            r"Operational $C_{L,\max}$",
        )

        ax.set_xlabel(r"$C_D$")
        ax.set_ylabel(r"$C_L$")
        ax.set_title(r"Drag Polar for Input $C_L$ Range")
        ax.grid(True)
        ax.legend()

        if created_figure:
            figure.tight_layout()

        return ax

    def plot_ld_vs_cl(self, ax=None):
        if self.results is None:
            self.compute()

        created_figure = ax is None

        if created_figure:
            figure, ax = plt.subplots(figsize=(7, 5))

        ax.plot(self.results["cl_values"], self.results["ld_values"], label=r"$L/D$")
        ax.scatter(
            self.results["cl_at_max_ld"],
            self.results["max_ld"],
            marker="o",
            color="tab:green",
            zorder=3,
            label=f"Max L/D = {self.results['max_ld']:.2f}",
        )

        if self.cl_min <= self.results["cl_cruise"] <= self.cl_max:
            ax.scatter(
                self.results["cl_cruise"],
                self.results["ld_cruise"],
                marker="s",
                color="tab:orange",
                zorder=3,
                label=f"Cruise L/D = {self.results['ld_cruise']:.2f}",
            )

        self._add_vertical_limit_if_visible(
            ax,
            self.results["cl_min_operational"],
            self.cl_min,
            self.cl_max,
            r"Operational $C_{L,\min}$",
        )
        self._add_vertical_limit_if_visible(
            ax,
            self.cl_max_operational,
            self.cl_min,
            self.cl_max,
            r"Operational $C_{L,\max}$",
        )

        ax.set_xlabel(r"$C_L$")
        ax.set_ylabel(r"$L/D$")
        ax.set_title(r"Lift-to-Drag Ratio for Input $C_L$ Range")
        ax.grid(True)
        ax.legend()

        if created_figure:
            figure.tight_layout()

        return ax

    def plot_all(self, show=True):
        self.power_curve.plot()
        self.plot_drag_polar()
        self.plot_ld_vs_cl()

        if show:
            plt.show()


def main():
    drag_polar_analysis = DragPolarAnalysis()
    drag_polar_analysis.compute()
    drag_polar_analysis.print_summary()
    drag_polar_analysis.plot_all(show=True)


if __name__ == "__main__":
    main()
