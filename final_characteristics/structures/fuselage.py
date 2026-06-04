import numpy as np
from parameters import *

class Cabin():
    def __init__(self, cabin_length, cabin_height, cabin_air_density, cabin_temperature):
        self.length = cabin_length
        self.height = cabin_height
        self.air_density = cabin_air_density
        self.temperature = cabin_temperature


class Material():

    def __init__(self, young_mod, yield_stress, density):
        self.young_mod = young_mod
        self.yield_stress = yield_stress
        self.density = density


def initial_skin_sizing(cabin, yield_stress, outer_air_pressure):


    safety_factor = 0.4
    cabin_air_pressure = cabin.air_density * R_AIR * cabin.temperature


    pressure_difference = cabin_air_pressure - outer_air_pressure

    acceptable_stress = yield_stress * safety_factor


    #hoop stress is the critical stress
    skin_thickness = pressure_difference * cabin.height/2 / acceptable_stress


    print(skin_thickness)

    # Approximating cabin as a capsule to calculate its surface area
    #capsule_area = cabin.length * 2 * np.pi * cabin.height / 2

cabin = Cabin(3.4, 2.3, 1.225, 23 + 273.15)

#initial_skin_sizing(cabin, 300e6, 63182)




### Forces and moments on hinges

def compute_FBD_wings():
    #theta, phi, thruster_position
    # Assumptions
    # Hinge lies at origin
    thruster_position = np.array([3, 0, 0.5])
    thruster_force = np.array([0, 0, 1000])
    hinge_orientation = np.array([1, 1, 1])

    print(compute_moment(thruster_force, thruster_position, hinge_orientation))

def compute_moment(force, force_position, axis):
    magn = np.sqrt(np.dot(force, force))
    print(magn)
    fdir = norm(force)
    adir = norm(axis)

    ax = norm(np.dot(fdir, adir)/(np.dot(adir, adir)) * adir)
    lat = fdir - ax
    moment_arm = np.cross(force_position, adir)/np.abs(adir)

    return(ax*magn, lat*magn, moment_arm)


compute_FBD_wings()