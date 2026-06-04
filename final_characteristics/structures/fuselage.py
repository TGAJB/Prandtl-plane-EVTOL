import numpy as np
import parameters as *


def initial_skin_sizing(cabin_length, cabin_height, yield_stress):

    safety_factor = 0.8

    acceptable_stress = yield_stress * safety_factor

    #Approximating cabin as a capsule to calculate its surface area
    capsule_area = cabin_length * 2 * np.pi * cabin_height/2

    cabin_pressure =


    #hoop stress is the critical stress
    skin_thickness = cabin_pressure * cabin_height/2 / acceptable_stress




