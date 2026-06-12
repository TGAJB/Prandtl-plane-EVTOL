
import aircraft
import math

def main():
    """ Compute the incidence angle of the aft wing """
    
    # Create an aircraft instance
    params = aircraft.AircraftParameters()
    physical = aircraft.Physical()
    fc = aircraft.FlightCondition()
    dc = aircraft.DatcomChartInputs()
    ac = aircraft.Aircraft(params, physical, fc, dc)

    # Variable definitions
    W             = ac.params.mass.mtow * 9.81
    rho           = ac.fc.rho
    S             = ac.params.wing_geometry.S_fw
    v             = ac.fc.tas
    CL_alpha      = ac.params.aerodynamics.CL_alpha_fw
    downwash_grad = ac.downwash_gradient()

    # Compute lift coefficient of the front wing
    CL = W / (rho * v**2 * S)
    alpha_eff = CL/CL_alpha

    # Compute the downwash angle. This is equal to the incidence angle.
    epsilon = downwash_grad * alpha_eff
    print(math.degrees(epsilon))


if __name__ == '__main__':
    main()