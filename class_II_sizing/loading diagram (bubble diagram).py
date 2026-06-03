import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import *
import matplotlib.pyplot as plt
import numpy as np

###Returns c.g range for aircraft

def bubble_diagram(mtow_kg, x_cg_oew):
    #Payload consists of 4 80kg passengers, 20kg each of cargo
    OEM = mtow_kg - M_PAYLOAD
    m_pass = 80
    m_cargo = 20 #cg of cargo assumed same as passengers
    x_pass_front = 2
    x_pass_back = 3

    massplots = []
    cgplots = []


    for i in range(3):
        plot = plot_passengers(OEM, x_cg_oew, m_pass*(1+0.05*i), m_cargo*(1+0.05*i), x_pass_front, x_pass_back)
        massplots.append(plot[0])
        cgplots.append((plot[1]))
        massplots.append(plot[2])
        cgplots.append((plot[3]))


    massplotsnp = np.array(massplots)
    cgplotsnp = np.array(cgplots)

    max_cg_idx = np.max(cgplotsnp)
    min_cg_idx = np.min(cgplotsnp)

    print(max_cg_idx)

    for i in range(len(cgplots)):
        plt.plot(cgplots[i], massplots[i], marker='o')

    plt.xlabel("center of Gravity [m]")
    plt.ylabel("Aircraft Mass [Kg]")
    plt.title("Bubble graph")
    plt.grid(True)
    plt.show()

def plot_passengers(starting_mass, starting_cg, m_pass, m_cargo, x_pass_front, x_pass_back):
    masslstfront = [starting_mass]
    masslstback = [starting_mass]
    cglstfront = [starting_cg]
    cglstback = [starting_cg]

    for i in range(2):
        add_mass(masslstfront, cglstfront, m_pass, x_pass_front)
        add_mass(masslstfront, cglstfront, m_cargo, x_pass_front)
        add_mass(masslstback, cglstback, m_pass, x_pass_back)
        add_mass(masslstback, cglstback, m_cargo, x_pass_back)

    for i in range(2):
        add_mass(masslstfront, cglstfront, m_pass, x_pass_back)
        add_mass(masslstfront, cglstfront, m_cargo, x_pass_back)
        add_mass(masslstback, cglstback, m_pass, x_pass_front)
        add_mass(masslstback, cglstback, m_cargo, x_pass_front)

    return masslstfront, cglstfront, masslstback, cglstback

def add_mass(masslst, cglst, mass, cg):
    cglst.append((masslst[-1]*cglst[-1] + mass * cg)/(masslst[-1] + mass))
    masslst.append(masslst[-1] + mass)
    return masslst, cglst



#bubble_diagram(1600, 2.5)
#print(add_mass([2], [0], 2, 1))
