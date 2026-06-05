import numpy as np
import matplotlib.pyplot as plt
from pandas.io.sas.sas_constants import header_size_length
from scipy.integrate import trapezoid
from scipy.integrate import cumulative_trapezoid
import pandas as pd
import parameters
from main import MTOW_FINAL as MTOW
from main import converged_mass
from mass_components import wing_mass
from mass_components import motor_mass
from mass_components import propeller_mass


# ============================================================
# Base Load Classes
# ============================================================

class Load:<<
    def plot(self, ax):
        raise NotImplementedError


class PointLoad(Load):
    def __init__(self, location, magnitude, label="Point Load"):
        """
        location : x-position along wing span [m]
        magnitude : force [N]
                    positive = upward
                    negative = downward
        """
        self.location = location
        self.magnitude = magnitude
        self.label = label

    def plot(self, ax):
        scale = 0.5
        dy = self.magnitude * scale

        ax.arrow(
            self.location,
            0,
            0,
            dy,
            width=0.04,
            head_width=0.2,
            head_length=max(abs(dy) * 0.25, 0.5),
            length_includes_head=True,
            color="red"
        )

        ax.text(
            self.location,
            self.magnitude * scale * 1.1,
            self.label,
            ha="center"
        )


class DistributedLoad(Load):
    def __init__(self, x_start, x_end, label="Distributed Load"):
        self.x_start = x_start
        self.x_end = x_end
        self.label = label

    def evaluate(self, x):
        raise NotImplementedError

    def resultant(self, n=1000):
        x = np.linspace(self.x_start, self.x_end, n)
        q = self.evaluate(x)
        return trapezoid(q, x)

    def plot(self, ax, n=200):
        x = np.linspace(self.x_start, self.x_end, n)
        q = self.evaluate(x)

        ax.fill_between(
            x,
            0,
            q,
            alpha=0.3,
            label=self.label
        )


class FunctionDistributedLoad(DistributedLoad):
    def __init__(self, x_start, x_end, function, label="Function Load"):
        super().__init__(x_start, x_end, label)
        self.function = function

    def evaluate(self, x):
        return self.function(x)


class CSVDistributedLoad(DistributedLoad):
    def __init__(self, csv_file, x_column, load_column,
                 label="CSV Load"):
        self.data = pd.read_csv(csv_file)

        self.x_data = self.data[x_column].values
        self.q_data = self.data[load_column].values

        super().__init__(
            self.x_data.min(),
            self.x_data.max(),
            label
        )

    def evaluate(self, x):
        return np.interp(x, self.x_data, self.q_data)


# ============================================================
# Wing Model
# ============================================================

class WingLoadDiagram:
    def __init__(self, span):
        self.span = span
        self.loads = []

    def add_load(self, load):
        self.loads.append(load)

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 5))

        # Wing reference line
        ax.plot([0, self.span], [0, 0],
                color='black', linewidth=3)

        for load in self.loads:
            load.plot(ax)

        ax.set_xlabel("Spanwise Position [m]")
        ax.set_ylabel("Load Intensity [N/m] or Point Force")
        ax.set_title("Aircraft Wing Load Diagram")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.show()


# ============================================================
# Example Aircraft
# ============================================================

# Half-span wing model
span = parameters.WING_SPAN / 2  # m

# Aircraft total weight supported by both wings
aircraft_weight = MTOW * 9.81 / 2 # N

# Half-wing must carry half the aircraft weight
required_lift = aircraft_weight / 2

wing_weight = wing_mass(MTOW) * 9.81 / 4.0  # N


# ============================================================
# Parabolic Lift Distribution
# q(x) = q0 * (1 - (x/L)^2)
# Choose q0 such that integral equals required_lift
# ============================================================

L = span

# Integral:
# ∫ q0(1-(x/L)^2) dx from 0 to L
# = q0 * 2L/3

q0 = required_lift / (2 * L / 3)

def lift_distribution(x):
    return q0 * (1 - (x / L)**2)


lift_load = FunctionDistributedLoad(
    0,
    L,
    lift_distribution,
    label="Lift"
)


# ============================================================
# Wing Self Weight
# Uniform downward load
# ============================================================

wing_weight_per_meter = -wing_weight / span

wing_weight_load = FunctionDistributedLoad(
    0,
    span,
    lambda x: wing_weight_per_meter * np.ones_like(x),
    label="Wing Weight"
)


# ============================================================
# Engine Loads
# Downward point forces
# ============================================================

engine1 = PointLoad(
    location=2.0,
    magnitude=-(converged_mass()["motors"] / 6 + converged_mass()["props"] + converged_mass()["hubs"]) * 9.81 ,
    label="Engine 1"
)

engine2 = PointLoad(
    location=4.0,
    magnitude=-(converged_mass()["motors"] / 6 + converged_mass()["props"] + converged_mass()["hubs"]) * 9.81,
    label="Engine 2"
)

def Vertical_force_diagram():
    # ============================================================
    # Assemble Diagram
    # ============================================================

    diagram = WingLoadDiagram(span)

    diagram.add_load(lift_load)
    diagram.add_load(wing_weight_load)
    diagram.add_load(engine1)
    diagram.add_load(engine2)

    # ============================================================
    # Check Force Balance
    # ============================================================

    lift_resultant = lift_load.resultant()
    wing_weight_resultant = wing_weight_load.resultant()

    point_load_sum = (
        engine1.magnitude +
        engine2.magnitude
    )

    print(f"Lift resultant:       {lift_resultant:.1f} N")
    print(f"Wing weight:          {wing_weight_resultant:.1f} N")
    print(f"Engine loads total:   {point_load_sum:.1f} N")

    net_force = (
        lift_resultant +
        wing_weight_resultant +
        point_load_sum
    )

    print(f"Net force:            {net_force:.1f} N")

    # ============================================================
    # Plot
    # ============================================================

    diagram.plot()


Vertical_force_diagram()