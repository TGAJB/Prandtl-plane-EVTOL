#Method based on values used with Khazan Ansat helicopter drop test, intrapolated to fit our model
#Source: https://dspace-erf.nlr.nl/server/api/core/bitstreams/a0a7be04-c625-476c-997e-25f5dc2b40a6/content
from parameters import *

mtow_ansat = 3600 #kg

#max loads bsaed off limit energy absorption (test)
max_load_fwd = 2436 * G #N
max_load_aft = 3202 * G #N
max_defl_fwd = 0.129 #m
max_defl_aft = 0.133 #m


###Ignore this
"""
work_fwd = 95 * G #Nm
work_aft = 104 * G #Nm
#Work done by the railings to stop the helicopter is equalled to its grav pot energy, used to find height at which it was dropped
total_work = work_fwd + work_aft
distance_dropped = total_work/(mtow_ansat*G)
print(distance_dropped)
"""

"""
def landing_gear_cross_sectional_area(Number_of_Railings, max_elastic_stress, young_mod, mtow_kg):

    scale_factor = mtow_kg/mtow_ansat

    scaled_load_fwd = max_load_fwd * scale_factor
    scaled_load_aft = max_load_aft * scale_factor
    scaled_defl_fwd = max_defl_fwd * scale_factor
    scaled_defl_aft = max_defl_aft * scale_factor


    #For now, it is assumed both forward and aft railings will experience similar stresses
    max_stress = scaled_defl_fwd * young_mod
    if max_stress > max_elastic_stress * 0.8: #0.8 is a guess here
        print("Material will deform plastically!")

    Total_Area = scaled_defl_fwd / max_stress
    Railing_Area = Total_Area / Number_of_Railings
    print(max_stress/10**6)
    print(Railing_Area)
"""
def landing_gear_cross_sectional_area(area, max_elastic_stress, young_mod, mtow_kg):

    scale_factor = mtow_kg/mtow_ansat

    scaled_load_fwd = max_load_fwd * scale_factor
    scaled_load_aft = max_load_aft * scale_factor
    scaled_defl_fwd = max_defl_fwd * scale_factor
    scaled_defl_aft = max_defl_aft * scale_factor

    max_stress = scaled_load_fwd/area
    print(max_stress)

    #For now, it is assumed both forward and aft railings will experience similar stresses
    max_stress = scaled_defl_fwd * young_mod
    if max_stress > max_elastic_stress * 0.8: #0.8 is a guess here
        print("Material will deform plastically!")

    Total_Area = scaled_defl_fwd / max_stress


landing_gear_cross_sectional_area(0.0035, 290e6, 73.1e9, 2000)