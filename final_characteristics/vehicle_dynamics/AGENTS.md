# Context for Codex: Prandtl Plane VTOL OEI CG-envelope script

We are designing a Prandtl Plane eVTOL in VTOL hover configuration.

The task is to create a Python script that determines the longitudinal CG-envelope for which static hover trim can be achieved when any one propeller/engine goes inoperative. The critical cases are most likely the cases where one of the most aft or forward placed propellers goes inoperative. Thus, only consider the cases where one propeller of the most aft or most forward cases goes inoperative.

## Physical assumptions

- The vehicle is in hover.
- Propellers produce vertical thrust only.
- No propeller can produce reverse thrust:
  \[
  0 \leq T_i
  \]
- Each propeller has a maximum thrust:
  \[
  T_i \leq T_{i,\max}
  \]
- One propeller at a time can fail:
  \[
  T_k = 0
  \]
- The total vertical thrust must equal weight:
  \[
  \sum_{i \neq k} T_i = W
  \]
- The lateral CG is assumed to be on the aircraft centreline:
  \[
  y_{CG}=0
  \]
- The aircraft is symmetric about the body longitudinal axis, so mirrored propeller failures give the same longitudinal CG envelope.

## Coordinate system

Use a Cartesian coordinate system:

- \(x\): longitudinal coordinate
- \(y\): lateral coordinate
- \(z\): vertical coordinate

The datum can be chosen arbitrarily, for example the nose, nominal CG, or geometric centre of the propeller layout. The only requirement is that all propeller coordinates and the CG coordinate use the same datum.

Each propeller has planform coordinates:

\[
(x_i, y_i)
\]

## Trim conditions

For failed propeller \(k\), the operating thrusts must satisfy:

\[
T_k = 0
\]

\[
\sum_{i\neq k} T_i = W
\]

\[
\sum_{i\neq k} T_i x_i = W x_{CG}
\]

\[
\sum_{i\neq k} T_i y_i = W y_{CG}
\]

Since \(y_{CG}=0\):

\[
\sum_{i\neq k} T_i y_i = 0
\]

and:

\[
0 \leq T_i \leq T_{i,\max}
\]

## Important note: thrust distribution is not unique

For one failed propeller, there are five operating thrusts but only three main static hover equilibrium equations:

\[
\sum T_i = W
\]

\[
\sum T_i x_i = W x_{CG}
\]

\[
\sum T_i y_i = W y_{CG}
\]

Therefore, the thrust distribution is generally not unique. With five unknown thrusts and three equations, there are usually two remaining degrees of freedom.

Because of this, the script should not try to solve a single unique thrust distribution from the equilibrium equations alone.

Instead, the script should treat the problem as a feasibility/optimization problem. For each failed propeller, it should use linear programming to find the operating thrust distribution that gives the minimum and maximum achievable longitudinal CG location while satisfying:

\[
\sum_{i\neq k} T_i = W
\]

\[
\sum_{i\neq k} T_i y_i = 0
\]

\[
0 \leq T_i \leq T_{i,\max}
\]

\[
T_k = 0
\]

The objective for the forward limit is:

\[
\min \sum_{i\neq k} T_i x_i
\]

and the objective for the aft limit is:

\[
\max \sum_{i\neq k} T_i x_i
\]

The corresponding CG location is then:

\[
x_{CG} = \frac{\sum_{i\neq k} T_i x_i}{W}
\]

The thrust vectors returned by the optimizer are not unique in general. They are only one feasible thrust distribution that achieves the requested CG limit.

## Longitudinal CG envelope

For each failed propeller \(k\), find the minimum and maximum trimmable longitudinal CG location by solving two linear programming problems.

Forward/aft limits:

\[
x_{CG,\min,k}
=
\min_{\mathbf{T}}
\frac{\sum_{i\neq k} T_i x_i}{W}
\]

\[
x_{CG,\max,k}
=
\max_{\mathbf{T}}
\frac{\sum_{i\neq k} T_i x_i}{W}
\]

subject to:

\[
\sum_{i\neq k} T_i = W
\]

\[
\sum_{i\neq k} T_i y_i = 0
\]

\[
T_k = 0
\]

\[
0 \leq T_i \leq T_{i,\max}
\]

The final OEI-safe longitudinal CG envelope is the intersection of all single-failure envelopes:

\[
x_{CG,\min}
=
\max_k x_{CG,\min,k}
\]

\[
x_{CG,\max}
=
\min_k x_{CG,\max,k}
\]

Because of symmetry, only one failure from each mirrored pair needs to be checked, but the script may also check all six propellers for verification.

## Desired Python implementation

Create a Python script that:

1. Takes as input:
    From the file 'vd_parameters.py' (Make sure to create an instance of the class AircraftParameters)
    - MTOW `mtow`
    - propeller coordinates `x_vtol_i`, `y_vtol_i`
      - The nose is taken as the datum to measure 'x_vtol_i' in [m]
      - The longitudinal axis (the x-axis) is the symmetry line, and the perpendicular distance to this line denotes the     y-coordinate
    - maximum thrust per propeller at sea level `T_max_SL` in [N]
   - optional list of failed propeller cases

2. For each failed propeller:
   - sets that propeller thrust to zero
   - solves a linear program to minimize `x_CG`
   - solves a linear program to maximize `x_CG`
   - returns the thrust distribution at each limit

3. Computes the final safe OEI longitudinal CG envelope by intersecting the individual failure envelopes.

4. Uses `scipy.optimize.linprog` if SciPy is available.

5. Includes clear printed output:
   - failed propeller number
   - minimum trimmable `x_CG` w.r.t. the coordinate system used during the derivation
   - maximum trimmable `x_CG` w.r.t. the coordinate system used during the derivation
   - thrust distribution at each limit in the form of forces in Newton per propeller
   - final intersection envelope w.r.t. the coordinate system used during the derivation
   - minimum trimmable `x_CG` w.r.t. the LEMAC of the front wing and normalized by the MAC of the front wing
   - maximum trimmable `x_CG` w.r.t. the LEMAC of the front wing and normalized by the MAC of the front wing
   - final intersection envelope w.r.t. the LEMAC of the front wing and normalized by the MAC of the front wing

6. Includes checks for infeasible cases.

7. Uses clear variable names and comments.