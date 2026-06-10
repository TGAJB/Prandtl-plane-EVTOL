"""
================================================================================
 PROPELLER SIZING & NOISE OPTIMIZER 
================================================================================

WHAT THIS DOES
  1. Sizes a variable-pitch propeller for HOVER and CRUISE independently:
       - finds the (RPM, collective pitch) that meets the required thrust
       - subject to a tip-Mach ceiling and a no-stall guard
       - choosing the MINIMUM-POWER feasible point.
  2. Runs the JPL TR 32-1462 far-field noise estimate at each sized point and
     checks it against a dB limit.
  3. Produces Figure-10.5-style plots:
       (a) C_T vs advance ratio J, swept over blade angle
       (b) C_T vs RPM,            swept over blade angle

AERODYNAMIC MODEL
  Simplified blade-element / momentum theory (BEMT):
    - HOVER : combined-inflow equation (Leishman form); FoM bounded <= 1.
    - CRUISE: forward-flight BEMT with iterated induced inflow.
    - Airfoil: tabulated Cl/Cd polar (placeholder included; swap for XFOIL data).
  This is a FIRST-ESTIMATE tool. Treat results above tip-Mach ~0.7 or near
  stall as qualitative, and the noise numbers as +/-10 dB (per the JPL report).

HOW TO USE
  Edit the PARAMETERS block below, then run:  python propeller_optimizer.py
  Outputs: console summary + two PNG figures.
================================================================================
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from energy import takeoff_power, vertical_climb_power, climb_acceleration_power, climb_power, cruise_power, landing_power, mission_energy
from parameters import T_TAKEOFF, T_CLIMB_ACC, T_CLIMB, T_CRUISE, T_LANDING, V_AVG_TO, V_CRUISE, V_HOVER, N_PROP, ETA_POWERTRAIN_HOVER, ETA_CLIMB, ETA_CRUISE, LD_CRUISE,G

# ==============================================================================
#  PARAMETERS  --  EDIT EVERYTHING IN THIS BLOCK
# ==============================================================================

mtow_kg = 2100 #CHANGE THIS for the actual MTOW in the end in    [kg]

# ---- Propeller geometry ------------------------------------------------------
DIAMETER       = 1.90  # propeller diameter (FIXED in your case)     [m]
N_BLADES       = 8    # number of blades (your free variable)
A_prop = np.pi * (DIAMETER/2)**2  # propeller disk area [m^2]
TWIST_ROOT_DEG = 14.0       # built-in geometric twist at the root        [deg]
TWIST_TIP_DEG  = 2.0        # built-in geometric twist at the tip         [deg]
CHORD_ROOT     = None       # root chord [m]; None -> auto (R/15)
CHORD_TIP      = None       # tip  chord [m]; None -> auto (R/25)
ROOT_CUTOUT    = 0.10      # fraction of R where the aerofoil begins
N_ELEMENTS     = 40         # spanwise blade elements (resolution)


# ---- Vehicle / mission (per-PROP thrust; divide totals by number of props) ---
N_PROPS        = 6         # nuxmber of propellers on the vehicle
T_HOVER_PROP   = mtow_kg * (G + V_HOVER / T_TAKEOFF) / N_PROPS  # thrust required per prop in hover           [N]
T_CRUISE_PROP  = mtow_kg * G /LD_CRUISE / N_PROPS  # thrust required per prop in cruise          [N]
V_CRUISE       = 55.5556    # cruise forward speed                        [m/s]
CRUISE_RPM_MIN = 800.0      # minimum cruise RPM (raises torque-expensive low-RPM corner)

# ---- Constraints / objective -------------------------------------------------
M_TIP_MAX      = 0.75      # tip Mach number ceiling (sets max RPM)
STALL_MARGIN   = 1.0        # keep max section AoA this many deg below stall
PITCH_MIN_DEG  = -2.0       # collective pitch search lower bound          [deg]
PITCH_MAX_DEG  = 60.0       # collective pitch search upper bound          [deg]
FOM_ASSUMED = 0.75 # Assumed figure of merit 

# ---- Environment -------------------------------------------------------------
RHO            = 1.225      # air density                                 [kg/m^3]
SOUND_SPEED    = 343.0      # speed of sound (aero)                       [m/s]
SOUND_SPEED_FT = 1125.0     # speed of sound for the noise model          [ft/s]

# ---- Noise check (JPL far-field) --------------------------------------------
NOISE_LIMIT_TO_DB = 80.0    # take-off noise requirement                  [dB]
NOISE_LIMIT_CR_DB = 60.0    # cruise   noise requirement                  [dB]
NOISE_OBS_DIST_M  = 15  # observer slant distance to a prop           [m]
NOISE_OBS_THETA   = 120  # angle from prop heading to observer         [deg]
OBS_THETA_CHANGE = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]  # for noise vs directivity plot

# ---- Figure 10.5-style plot sweep -------------------------------------------
PLOT_BLADE_ANGLES_DEG = [25, 30, 35, 40, 45, 50, 55, 60, 65]  # TOTAL blade angle
PLOT_RPM_FIXED        = 600    # RPM held fixed for the C_T-vs-J plot
PLOT_V_FIXED          = 55.0     # forward speed held fixed for C_T-vs-RPM [m/s]
PLOT_J_RANGE          = (0.0, 4.2)
PLOT_RPM_RANGE        = (300, 4500)
PLOT_REF_STATION      = 0.75     # r/R at which "blade angle" is defined

# ---- Unit conversions (do not edit) -----------------------------------------
FT_PER_M = 3.28084
LBF_PER_N = 0.224809
HP_PER_W = 1.0 / 745.7


# ==============================================================================
#  AIRFOIL POLAR  (placeholder NACA-23012-ish; replace with XFOIL CSV data)
# ==============================================================================
# Tabulated Cl, Cd vs alpha [deg]. CLARK-Y at Re ~ 0.5e6 (matches the cruise
# Reynolds number at the 0.75R station for a slender-to-medium blade).
# Representative values for first analysis -- replace with XFOIL at your EXACT
# Reynolds number once chord/solidity is fixed. Low-Re => lower Cl_max (~1.3),
# higher drag (Cd_min ~0.0115), softer stall than the high-Re (1e6) version.
_POLAR_ALPHA = np.array([-8,-6,-4,-2,0,2,4,6,8,10,12,13,14,16,18,20,25,30])
_POLAR_CL    = np.array([-0.4,-0.22,-0.03,0.16,0.36,0.56,0.75,0.93,1.08,1.2,
                         1.28,1.3,1.28,1.15,1.0,0.92,0.88,0.92])
_POLAR_CD    = np.array([0.024,0.018,0.014,0.012,0.0115,0.012,0.0135,0.016,
                         0.021,0.028,0.04,0.048,0.06,0.11,0.17,0.23,0.33,0.45])

fig, ax1 = plt.subplots()
ax1.scatter(_POLAR_ALPHA, _POLAR_CL, label="Cl", color="tab:blue")
ax1.set_xlabel("alpha [deg]"); ax1.set_ylabel("Cl")
ax2 = ax1.twinx()
ax2.scatter(_POLAR_ALPHA, _POLAR_CD, label="Cd", color="tab:orange")
ax2.set_ylabel("Cd")
fig.legend()
plt.savefig("polar.png", dpi=150)

class Polar:
    """Linear-interpolated airfoil polar with stall detection."""
    def __init__(self, alpha_deg, cl, cd):
        a = np.deg2rad(alpha_deg)
        o = np.argsort(a)
        self.a, self.cl_t, self.cd_t = a[o], cl[o], cd[o]
        pos = self.a >= 0
        i = np.argmax(self.cl_t[pos])
        self.alpha_stall = self.a[pos][i]      # rad, positive-side Cl_max angle
        self.cl_max = self.cl_t[pos][i]
    def cl(self, ar): return float(np.interp(ar, self.a, self.cl_t))
    def cd(self, ar): return float(np.interp(ar, self.a, self.cd_t))
    def slope(self):                            # lift-curve slope near 0..5 deg
        return (self.cl(np.deg2rad(5)) - self.cl(0.0)) / np.deg2rad(5)

POLAR = Polar(_POLAR_ALPHA, _POLAR_CL, _POLAR_CD)


# ==============================================================================
#  BLADE GEOMETRY
# ==============================================================================
class Blade:
    """Discretized blade: radial stations, chord, built-in twist."""
    def __init__(self):
        self.R = DIAMETER / 2.0
        self.B = N_BLADES
        cr = CHORD_ROOT if CHORD_ROOT else self.R / 15.0
        ct = CHORD_TIP  if CHORD_TIP  else self.R / 25.0
        edges = np.linspace(ROOT_CUTOUT * self.R, self.R, N_ELEMENTS + 1)
        self.r = 0.5 * (edges[:-1] + edges[1:])      # element centers
        self.dr = np.diff(edges)
        x = (self.r - ROOT_CUTOUT * self.R) / (self.R - ROOT_CUTOUT * self.R)
        self.chord = cr + (ct - cr) * x
        self.twist = np.deg2rad(TWIST_ROOT_DEG +
                                (TWIST_TIP_DEG - TWIST_ROOT_DEG) * x)
        self.A = np.pi * self.R ** 2
    def twist_at(self, r_over_R):
        """Built-in twist [deg] at a given r/R (for plot labeling)."""
        x = (r_over_R - ROOT_CUTOUT) / (1 - ROOT_CUTOUT)
        return TWIST_ROOT_DEG + (TWIST_TIP_DEG - TWIST_ROOT_DEG) * x


# ==============================================================================
#  BEMT  --  CRUISE (forward flight)
# ==============================================================================
def bemt_cruise(blade, n_rpm, V_axial, collective_deg, rho=RHO):
    """
    Forward-flight BEMT with iterated induced inflow per annulus.
    Returns thrust, power, coefficients, advance ratio, efficiency, max AoA.
    """
    Omega = n_rpm * 2*np.pi / 60.0
    R = blade.R
    beta = blade.twist + np.deg2rad(collective_deg)   # section pitch
    r, Ut = blade.r, Omega * blade.r
    dT = np.zeros_like(r); dQ = np.zeros_like(r); amax = -np.inf

    for i in range(len(r)):
        a_i, b_i, alpha = 0.0, 0.0, 0.0
        for _ in range(100):                          # inflow iteration
            Up = V_axial * (1 + a_i)                  # axial velocity at disc
            Ur = Ut[i] * (1 - b_i)                    # tangential velocity
            phi = np.arctan2(Up, Ur)                  # inflow angle
            alpha = beta[i] - phi                     # angle of attack
            Cl, Cd = POLAR.cl(alpha), POLAR.cd(alpha)
            W2 = Up**2 + Ur**2
            # Prandtl tip loss
            f = blade.B/2 * (R - r[i]) / (r[i] * max(np.sin(phi), 1e-3))
            F = max((2/np.pi) * np.arccos(np.exp(-abs(f))), 1e-3)
            cphi, sphi = np.cos(phi), np.sin(phi)
            dL = 0.5*rho*W2*blade.chord[i]*Cl
            dD = 0.5*rho*W2*blade.chord[i]*Cd
            dTi = blade.B*(dL*cphi - dD*sphi)
            dQi = blade.B*(dL*sphi + dD*cphi)*r[i]
            # momentum update of induced factors
            if V_axial > 1e-3:
                vi = dTi/(4*np.pi*r[i]*rho*Up*F*blade.dr[i]+1e-12)*blade.dr[i]
                a_new = vi/V_axial
            else:
                a_new = 0.0
            b_new = (dQi/(4*np.pi*r[i]**3*rho*(V_axial*(1+a_i)+1e-9)*Omega*F
                     *blade.dr[i]+1e-12)*blade.dr[i]) if V_axial > 1e-3 else b_i
            if abs(a_new-a_i) < 1e-6 and abs(b_new-b_i) < 1e-6:
                a_i, b_i = a_new, b_new; break
            a_i = 0.5*a_i + 0.5*a_new
            b_i = 0.5*b_i + 0.5*b_new
        amax = max(amax, alpha)
        dT[i], dQ[i] = dTi, dQi

    T = np.sum(dT*blade.dr); Q = np.sum(dQ*blade.dr); P = Q*Omega
    n_rev, D = n_rpm/60.0, 2*R
    CT = T/(rho*n_rev**2*D**4) if n_rev > 0 else 0.0
    CP = P/(rho*n_rev**3*D**5) if n_rev > 0 else 0.0
    J  = V_axial/(n_rev*D) if n_rev > 0 else 0.0
    eta = (CT*J/CP) if (CP > 0 and J > 0) else 0.0
    return dict(T=T, P=P,CT=CT, CP=CP, J=J, eta=eta,
                Vtip=Omega*R, n_rpm=n_rpm, alpha_max=amax)


# ==============================================================================
#  BEMT  --  HOVER (combined-inflow equation, bounded FoM)
# ==============================================================================
def bemt_hover(blade, n_rpm, collective_deg, rho=RHO):
    """
    Hover via the standard combined blade-element/momentum inflow equation:
        lambda = sqrt(k^2 + s*a/8 * theta * r_hat) - k,   k = s*a/16
    FoM is bounded <= 1 by construction. Returns thrust, power, FoM, max AoA.
    """
    Omega = n_rpm * 2*np.pi / 60.0
    R = blade.R
    r_hat = blade.r / R
    dr_hat = blade.dr / R
    a_slope = POLAR.slope()
    theta = blade.twist + np.deg2rad(collective_deg)          # local pitch [rad]
    s = blade.B * blade.chord / (np.pi * R)                   # local solidity

    k = s * a_slope / 16.0
    lam = np.sqrt(k**2 + s*a_slope/8.0 * np.maximum(theta, 0) * r_hat) - k
    lam = np.maximum(lam, 1e-6)
    # tip-loss inflow correction
    f = blade.B/2 * (1 - r_hat) / np.maximum(lam, 1e-3)
    F = np.maximum((2/np.pi)*np.arccos(np.exp(-np.abs(f))), 1e-3)
    lam = lam / np.sqrt(F)

    phi = np.arctan2(lam, r_hat)
    alpha = theta - phi
    Cl = np.array([POLAR.cl(a) for a in alpha])
    Cd = np.array([POLAR.cd(a) for a in alpha])

    dCT = 0.5 * s * Cl * r_hat**2 * dr_hat
    dCP = 0.5 * s * (phi*Cl + Cd) * r_hat**3 * dr_hat
    CT, CP = np.sum(dCT), np.sum(dCP)
    VtipR = Omega * R
    T = CT * rho * blade.A * VtipR**2

    # --- Two hover-power estimates ---
    # (1) BEMT power, integrated from the blade aerodynamics. Optimistic here
    #     because the placeholder polar has low drag, so its implied FoM runs high.
    P_bemt = CP * rho * blade.A * VtipR**3
    FoM_bemt = T**1.5 / (np.sqrt(2*rho*blade.A) * P_bemt) if P_bemt > 0 else 0.0

    # (2) Momentum-theory power with an ASSUMED, literature-typical FoM.
    #     P_ideal = T^1.5 / sqrt(2*rho*A) is the lossless minimum (exact from
    #     momentum theory); dividing by FOM_ASSUMED imposes realistic losses
    #     instead of trusting the BEMT drag. This is the spreadsheet approach.
    P_ideal = T**1.5 / np.sqrt(2*rho*blade.A) if T > 0 else 0.0
    P_mom = P_ideal / FOM_ASSUMED if FOM_ASSUMED > 0 else 0.0

    # Use the momentum+assumed-FoM power as the design power (more realistic);
    # keep the BEMT value available for comparison.
    P = P_mom
    FoM = FOM_ASSUMED

    n_rev, D = n_rpm/60.0, 2*R
    CT_prop = T/(rho*n_rev**2*D**4) if n_rev > 0 else 0.0
    return dict(T=T, P=P, P_bemt=P_bemt, FoM=FoM, FoM_bemt=FoM_bemt,
                CT=CT_prop, Vtip=VtipR, n_rpm=n_rpm,
                alpha_max=float(np.max(alpha)))


# ==============================================================================
#  PITCH SOLVERS  (find collective pitch that meets a target thrust)
# ==============================================================================
def _solve_pitch(thrust_fn, thrust_req, p_lo, p_hi):
    """Root-find pitch on the PRE-STALL (rising) branch of thrust-vs-pitch."""
    scan = np.linspace(p_lo, p_hi, 60)
    Ts = np.array([thrust_fn(p) for p in scan])
    i_peak = int(np.argmax(Ts))                  # thrust peaks near stall
    if Ts[i_peak] < thrust_req:
        return None                              # unreachable before stall
    a, b = scan[0], scan[i_peak]                 # search only the rising side
    f = lambda p: thrust_fn(p) - thrust_req
    if f(a) > 0: return a
    if f(b) < 0: return None
    return brentq(f, a, b, xtol=0.01)


# ==============================================================================
#  MIN-POWER SIZING FOR ONE PHASE
# ==============================================================================
def size_phase(blade, thrust_req, V_axial, hover):
    """
    Walk RPM up to the tip-Mach ceiling; at each RPM solve pitch for thrust,
    apply guards (thrust met, no stall, pitch band), keep the min-power point.
    Returns (best, n_feasible, n_rejected).
    """
    rpm_ceiling = M_TIP_MAX * SOUND_SPEED / blade.R * 60.0 / (2*np.pi)
    alpha_limit = POLAR.alpha_stall - np.deg2rad(STALL_MARGIN)
    rpm_floor = 200.0 if hover else CRUISE_RPM_MIN
    rpms = np.linspace(rpm_floor, rpm_ceiling, 140)
    feasible, rejected = [], 0

    for n in rpms:
        if hover:
            tf = lambda p: bemt_hover(blade, n, p)["T"]
        else:
            tf = lambda p: bemt_cruise(blade, n, V_axial, p)["T"]
        p = _solve_pitch(tf, thrust_req, PITCH_MIN_DEG, PITCH_MAX_DEG)
        if p is None:
            continue
        res = bemt_hover(blade, n, p) if hover else bemt_cruise(blade, n, V_axial, p)
        res["pitch_deg"] = p
        res["M_tip"] = res["Vtip"] / SOUND_SPEED
        # ---- guards ----
        if abs(res["T"] - thrust_req) > 0.02*thrust_req: rejected += 1; continue
        if res["alpha_max"] > alpha_limit:               rejected += 1; continue
        if not (PITCH_MIN_DEG <= p <= PITCH_MAX_DEG):     rejected += 1; continue
        if hover and res["FoM"] > 1.0:                    rejected += 1; continue
        feasible.append(res)

    best = min(feasible, key=lambda r: r["P"]) if feasible else None
    return best, len(feasible), rejected


# ==============================================================================
#  NOISE CHECK  (JPL TR 32-1462 far-field, see report Appendix B Section C)
# ==============================================================================
def _L1_ref(hp):           # Fig. B-2 reference level (DIGITIZED — verify)
    return 121.0 + 14.0*np.log10(hp/300.0)
_MT = np.array([0.2,0.3,0.4,0.5,0.6,0.66,0.7,0.8,0.9,1.0])
_L2 = np.array([-7,-5,-3.5,-2,-1.2,-1.0,-0.6,0.5,1.5,3.0])     # at Z/D=0.1
_ZD = np.array([0.01,0.02,0.04,0.1,0.2,0.4,1.0,2.0,4.0,10.0])
_ZS = np.array([18,13,7,0,-8,-18,-40,-55,-70,-90])
_TH = np.array([20,40,60,80,90,100,110,120,130,140,160,180])
_DR = np.array([-13,-6,-2,-0.5,0,2,3.5,4,3,1,-6,-12])
def _L2_corr(mt, ZD):      # Fig. B-3 (DIGITIZED — verify)
    return np.interp(mt, _MT, _L2) + np.interp(ZD, _ZD, _ZS)
def _Ldir(theta):          # Fig. B-8 average directivity (DIGITIZED — verify)
    return np.interp(theta, _TH, _DR)

def noise_spl(res, blade, r_ft, theta_deg):
    """Overall far-field SPL [dB] from N_PROPS identical propellers."""
    D_ft = 2*blade.R*FT_PER_M
    hp = res["P"]*HP_PER_W
    Mt = res["Vtip"]/SOUND_SPEED * (SOUND_SPEED/SOUND_SPEED)   # tip Mach
    Mt = (res["Vtip"]*FT_PER_M)/SOUND_SPEED_FT
    L1 = _L1_ref(hp)
    spl_each = (L1 + 20*np.log10(4.0/blade.B) + 40*np.log10(15.5/D_ft)
                + _L2_corr(Mt, 1.0/D_ft) + _Ldir(theta_deg)
                - 20*np.log10(r_ft - 1.0))
    return 10*np.log10(N_PROPS * 10**(spl_each/10.0))

# ==============================================================================
#  NOISE MODEL 2 -- A-weighted SOUND POWER LEVEL (PWL) regression
#  Source: Wang, Lima Pereira & Ragni, "Design exploration of UAM vehicles",
#  Aerospace Science and Technology 160 (2025) 110058, Eqs. (12)-(14).
#
#  Computes the A-weighted PWL [dBA] of the whole vehicle from basic powertrain
#  data. NOTE: this is sound POWER (PWL), not SPL at an observer distance, so it
#  is NOT directly comparable to the JPL noise_spl() output. It already includes
#  A-weighting, so do NOT pass it through spl_to_dba().
#
#  Fitted regression (per-stage coefficients, Eqs. 13 & 14):
#    Takeoff: PWL = 3.2*log10(Ps) - 20.3*log10(D) + 60*log10(Mt)
#                   - 2.9*B + 10*log10(Nprop) + 124.1
#    Cruise : PWL =              + 14.1*log10(D) + 60*log10(Mt)
#                   + 0.5*B + 10*log10(Nprop) + 102.2     (Ps term ~ 0)
# ==============================================================================
def pwl_regression(res, blade, stage, fom=None, n_props=None, c_sound=SOUND_SPEED):
    """
    A-weighted sound power level [dBA] via the Wang et al. (2025) regression.

    res     : a bemt_hover() or bemt_cruise() result dict (needs P, Vtip, n_rpm)
    blade   : the Blade object (for diameter and blade count B)
    stage   : "takeoff" or "cruise" -- selects the fitted coefficient set
    fom     : figure of merit used to convert shaft power to Ps = P/FM.
              Defaults to FOM_ASSUMED if available, else res['FoM'], else 0.75.
    n_props : number of propellers; defaults to N_PROPS.
    Returns the A-weighted PWL in dBA.
    """
    if n_props is None:
        n_props = N_PROPS
    if fom is None:
        fom = globals().get("FOM_ASSUMED", res.get("FoM", 0.75))

    D = 2 * blade.R                       # propeller diameter [m]
    B = blade.B                           # number of blades
    Ps = res["P"] / fom                   # shaft power per prop incl. FoM [W]
    Mt = res["Vtip"] / c_sound            # tip Mach number

    if Mt <= 0 or Ps <= 0:
        return float("nan")

    if stage == "takeoff":
        pwl = (3.2 * np.log10(Ps)
               - 20.3 * np.log10(D)
               + 60.0 * np.log10(Mt)
               - 2.9 * B
               + 10.0 * np.log10(n_props)
               + 124.1)
    elif stage == "cruise":
        pwl = (14.1 * np.log10(D)
               + 60.0 * np.log10(Mt)
               + 0.5 * B
               + 10.0 * np.log10(n_props)
               + 102.2)
    else:
        raise ValueError("stage must be 'takeoff' or 'cruise'")

    return pwl
def pwl_to_spl_directional(pwl_dba, r_m, theta_deg, hemisphere=True):
    """
    Convert A-weighted PWL [dBA] to SPL [dBA] at a SPECIFIC observer angle,
    by spreading the power over a hemisphere AND applying a zero-mean
    directivity index derived from the JPL Fig. B-8 curve (_Ldir).

    This is more accurate than the direction-averaged pwl_to_spl() because it
    accounts for the propeller radiating more strongly near the disc plane and
    less along the axis -- redistributing the fixed total power by angle.

    theta_deg : observer angle from the propeller heading (e.g. 100 deg).
    """
    area = 2*np.pi*r_m**2 if hemisphere else 4*np.pi*r_m**2
    spl_avg = pwl_dba - 10*np.log10(area)

    # build a zero-mean directivity index from the JPL Fig. B-8 curve.
    # energy-average _Ldir over the hemisphere (20..180 deg), then subtract.
    th = np.linspace(20, 180, 100)
    Ld = np.array([_Ldir(t) for t in th])
    # energy (intensity) average, weighted by sin(theta) for the solid angle
    w = np.sin(np.deg2rad(th))
    Ld_mean = 10*np.log10(np.sum(w * 10**(Ld/10.0)) / np.sum(w))
    D_theta = _Ldir(theta_deg) - Ld_mean

    return spl_avg + D_theta

# ------------------------------------------------------------------------------
#  dB -> dBA CONVERSION  (A-weighting via octave-band spectrum)
# ------------------------------------------------------------------------------
# dBA is NOT a fixed offset from dB: it weights each frequency band by human
# hearing sensitivity. We must (1) build the octave-band spectrum, (2) apply the
# A-weight per band, (3) logarithmically recombine. The spectrum comes from the
# JPL harmonic-distribution method (Fig. B-6): the overall SPL is distributed
# over harmonics of the blade-passage frequency, which are then grouped into
# octave bands.

# Standard A-weighting offsets [dB] at preferred octave-band centers [Hz].
_AW_FREQ = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
_AW_GAIN = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1])

# JPL Fig. B-6 harmonic level relative to overall SPL [dB], vs harmonic number
# of blade-passage frequency. DIGITIZED (the M_h~0.65-0.8 curve family) -- the
# fundamental is strongest, higher harmonics fall off. Verify against the chart.
_HARM_NUM = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
_HARM_REL = np.array([-2, -9, -13, -16, -18, -19, -20, -20, -20, -20])

def _octave_band_for(freq_hz):
    """Return the index of the preferred octave band containing freq_hz."""
    # band edges around the centers (factor sqrt(2) each side)
    for i, fc in enumerate(_AW_FREQ):
        lo, hi = fc/np.sqrt(2), fc*np.sqrt(2)
        if lo <= freq_hz < hi:
            return i
    if freq_hz < _AW_FREQ[0]/np.sqrt(2):
        return 0
    return len(_AW_FREQ) - 1

def spl_to_dba(res, blade, overall_spl_db):
    """
    Convert an overall SPL [dB] to dBA using the propeller's harmonic spectrum.

    Steps:
      1. blade-passage fundamental f1 = B * RPM / 60.
      2. distribute overall SPL over harmonics m*f1 using Fig. B-6 offsets.
      3. bin harmonic levels into octave bands (energy sum within each band).
      4. apply A-weighting per band, recombine to a single dBA.
    Returns (dba, band_table) where band_table lists (fc, SPL_band, SPL_A_band).
    """
    f1 = blade.B * res["n_rpm"] / 60.0           # blade-passage fundamental [Hz]

    # harmonic SPLs (relative to overall, then we re-normalize so the energy
    # sum of harmonics equals the given overall SPL)
    harm_freqs = _HARM_NUM * f1
    harm_rel = _HARM_REL                          # dB re overall
    # provisional harmonic SPLs (overall + relative); normalize afterwards
    prov = overall_spl_db + harm_rel
    # renormalize so 10log10(sum 10^(L/10)) == overall_spl_db
    e_sum = np.sum(10**(prov/10.0))
    norm = overall_spl_db - 10*np.log10(e_sum)
    harm_spl = prov + norm

    # bin into octave bands (sum energy of harmonics falling in each band)
    band_energy = np.zeros(len(_AW_FREQ))
    for f, L in zip(harm_freqs, harm_spl):
        b = _octave_band_for(f)
        band_energy[b] += 10**(L/10.0)
    band_spl = np.where(band_energy > 0, 10*np.log10(band_energy + 1e-30), -np.inf)

    # apply A-weighting and recombine
    band_spl_A = band_spl + _AW_GAIN
    dba = 10*np.log10(np.sum(10**(band_spl_A[np.isfinite(band_spl_A)]/10.0)))

    table = [(int(fc), float(s), float(sa))
             for fc, s, sa in zip(_AW_FREQ, band_spl, band_spl_A)
             if np.isfinite(s)]
    return dba, table


# ==============================================================================
#  FIGURE 10.5-STYLE PLOTS  (C_T vs J  and  C_T vs RPM, swept over blade angle)
# ==============================================================================
def make_figure(blade):
    """
    Reproduces the style of Figure 10.5: C_T vs advance ratio and vs RPM,
    one curve per TOTAL blade angle (beta at r/R = PLOT_REF_STATION).
    Collective is set so that twist(0.75R) + collective = beta.
    """
    twist_ref = blade.twist_at(PLOT_REF_STATION)        # deg at ref station
    Js   = np.linspace(*PLOT_J_RANGE, 40)
    rpms = np.linspace(*PLOT_RPM_RANGE, 40)
    n_rev_fixed = PLOT_RPM_FIXED / 60.0
    D = 2*blade.R

    fig, (axJ, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap = plt.cm.turbo(np.linspace(0, 1, len(PLOT_BLADE_ANGLES_DEG)))

    for beta, col in zip(PLOT_BLADE_ANGLES_DEG, cmap):
        collective = beta - twist_ref                   # back out collective
        # --- C_T vs J (fixed RPM, vary forward speed) ---
        CTj = []
        for J in Js:
            V = J * n_rev_fixed * D
            CTj.append(bemt_cruise(blade, PLOT_RPM_FIXED, V, collective)["CT"])
        axJ.plot(Js, CTj, color=col, label=f"$\\beta$={beta}\u00b0")
        # --- C_T vs RPM (fixed forward speed, vary RPM) ---
        CTr = []
        for n in rpms:
            CTr.append(bemt_cruise(blade, n, PLOT_V_FIXED, collective)["CT"])
        axR.plot(rpms, CTr, color=col, label=f"$\\beta$={beta}\u00b0")

    axJ.set_xlabel("$J = V/nD$"); axJ.set_ylabel("$C_T$")
    axJ.set_title(f"(a) $C_T$ vs advance ratio  (n={PLOT_RPM_FIXED} rpm)")
    axJ.grid(alpha=0.3); axJ.legend(fontsize=8, ncol=1)
    axJ.set_ylim(0.0, 0.25)

    axR.set_xlabel("RPM"); axR.set_ylabel("$C_T$")
    axR.set_title(f"(b) $C_T$ vs RPM  (V={PLOT_V_FIXED} m/s)")
    axR.grid(alpha=0.3); axR.legend(fontsize=8, ncol=1)
    axR.set_ylim(0.0, 0.25)

    fig.suptitle(f"Thrust coefficient — {blade.B}-blade propeller "
                 f"(D={DIAMETER} m), blade angle at r/R={PLOT_REF_STATION}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig("propeller_CT_curves.png", dpi=150)
    print("Saved: propeller_CT_curves.png")

# ==============================================================================
#  EFFICIENCY / PITCH-SELECTION DIAGRAM
# ==============================================================================
# This is the plot you ACTUALLY use to choose the cruise pitch: for each blade
# angle, it finds the RPM that meets the cruise thrust under the stall and
# tip-Mach constraints, and plots the resulting propulsive efficiency. The peak
# (among feasible points) is the design pitch -- unlike the C_T plot, which
# shows thrust, not efficiency, and so cannot be used to pick pitch directly.
def make_efficiency_figure(blade, betas=None):
    """
    Plot cruise propulsive efficiency vs blade angle (beta at 0.75R), with
    infeasible angles (stall or tip-Mach violation) marked. Also returns the
    best feasible (beta, eta, RPM, pitch) for reporting.
    """
    if betas is None:
        betas = np.arange(30, 66, 2.5)
    tw = blade.twist_at(PLOT_REF_STATION)
    T = T_CRUISE_PROP
    rpm_ceiling = M_TIP_MAX * SOUND_SPEED / blade.R * 60.0 / (2*np.pi)
    alpha_lim = POLAR.alpha_stall - np.deg2rad(STALL_MARGIN)
 
    betas_ok, eta_ok, rpm_ok = [], [], []
    betas_bad = []
    for beta in betas:
        coll = beta - tw
        ns = np.linspace(300, rpm_ceiling, 120)
        best = None
        for n in ns:
            r = bemt_cruise(blade, n, V_CRUISE, coll)
            mtip = r["Vtip"] / SOUND_SPEED
            if abs(r["T"] - T) < 0.03*T and r["alpha_max"] <= alpha_lim and mtip <= M_TIP_MAX:
                if best is None or r["eta"] > best[1]:   # keep highest eta
                    best = (beta, r["eta"], r["n_rpm"], coll)
        if best:
            betas_ok.append(best[0]); eta_ok.append(best[1]); rpm_ok.append(best[2])
        else:
            betas_bad.append(beta)
 
    fig, ax = plt.subplots(figsize=(8, 5))
    if betas_ok:
        ax.plot(betas_ok, eta_ok, "-o", color="tab:blue", label="feasible")
        i_best = int(np.argmax(eta_ok))
        ax.plot(betas_ok[i_best], eta_ok[i_best], "r*", ms=18,
                label=f"optimum $\\beta$={betas_ok[i_best]:.1f}\u00b0, "
                      f"$\\eta$={eta_ok[i_best]:.3f}")
    for bb in betas_bad:
        ax.axvline(bb, color="0.85", lw=6, zorder=0)
    ax.set_xlabel("Blade angle $\\beta$ at r/R=0.75 [deg]")
    ax.set_ylabel("Cruise propulsive efficiency  $\\eta$")
    ax.set_title("Cruise efficiency vs blade angle\n"
                 "(grey bands = infeasible: stall or tip-Mach)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig("propeller_efficiency_vs_pitch.png", dpi=150)
    print("Saved: propeller_efficiency_vs_pitch.png")
 
    if betas_ok:
        i = int(np.argmax(eta_ok))
        return dict(beta=betas_ok[i], eta=eta_ok[i], rpm=rpm_ok[i],
                    collective=betas_ok[i]-tw)
    return None
 
# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    blade = Blade()
    DL = T_HOVER_PROP / blade.A
    print("="*64)
    print(f" Propeller: D={DIAMETER} m, B={N_BLADES}, "
          f"disc loading {DL:.0f} N/m^2 ({DL/9.81:.0f} kg/m^2)")
    print(f" Airfoil stall at {np.rad2deg(POLAR.alpha_stall):.0f} deg, "
          f"Cl_max={POLAR.cl_max:.2f}")
    print("="*64)
 
    # ---------- HOVER ----------
    print("\n--- HOVER (min power) ---")
    hov, nf, nr = size_phase(blade, T_HOVER_PROP, 0.0, hover=True)
    if hov is None:
        print(f"  INFEASIBLE under M_tip<={M_TIP_MAX} ({nr} candidates rejected).")
        print("  -> increase diameter / blade count, or relax M_tip / noise.")
    else:
        spl = noise_spl(hov, blade, NOISE_OBS_DIST_M*FT_PER_M, NOISE_OBS_THETA)
        print(f"  RPM          : {hov['n_rpm']:8.1f}")
        print(f"  collective   : {hov['pitch_deg']:8.2f} deg")
        print(f"  thrust       : {hov['T']:8.1f} N  (target {T_HOVER_PROP:.0f})")
        print(f"  power        : {hov['P']/1000:8.2f} kW, BEMT power: {hov['P_bemt']/1000:8.2f} kW")
        print(f"  M_tip        : {hov['M_tip']:8.3f}")
        print(f"  FoM          : {hov['FoM']:8.3f}, BEMT FoM: {hov['FoM_bemt']:8.3f}")
        print(f"  max AoA      : {np.rad2deg(hov['alpha_max']):8.2f} deg "
              f"(stall {np.rad2deg(POLAR.alpha_stall):.0f})")
        print(f"Maximum torque exerted on motor shaft: {hov['P']*60/((hov['n_rpm'])*2*np.pi):.1f} Nm")
        v = "OK" if spl <= NOISE_LIMIT_TO_DB else f"EXCEEDS by {spl-NOISE_LIMIT_TO_DB:.1f}"
        print(f"  TO noise     : {spl:8.1f} dB @ {NOISE_OBS_DIST_M:.0f} m "
              f"(limit {NOISE_LIMIT_TO_DB}: {v})")
        dba, _ = spl_to_dba(hov, blade, spl)
        print(f"  TO noise     : {dba:8.1f} dBA (A-weighted)")
        pwl_t = pwl_regression(hov, blade, "takeoff")
        spl_t_dir = pwl_to_spl_directional(pwl_t, NOISE_OBS_DIST_M,
                                           NOISE_OBS_THETA, hemisphere=True)
        print(f"  TO PWL (Wang2025): {pwl_t:8.1f} dBA (sound power level)")
        print(f"  TO SPL (Wang2025, theta={NOISE_OBS_THETA:.0f} deg): "
              f"{spl_t_dir:8.1f} dBA")
    # ---------- CRUISE ----------
    print("\n--- CRUISE (min power) ---")
    cru, nf, nr = size_phase(blade, T_CRUISE_PROP, V_CRUISE, hover=False)
    if cru is None:
        print(f"  INFEASIBLE under M_tip<={M_TIP_MAX} ({nr} candidates rejected).")
    else:
        spl = noise_spl(cru, blade, NOISE_OBS_DIST_M*FT_PER_M, NOISE_OBS_THETA)
        print(f"  RPM          : {cru['n_rpm']:8.1f}")
        print(f"  collective   : {cru['pitch_deg']:8.2f} deg")
        print(f"  thrust       : {cru['T']:8.1f} N  (target {T_CRUISE_PROP:.0f})")
        print(f"  power        : {cru['P']/1000:8.2f} kW")
        print(f"  J            : {cru['J']:8.3f}")
        print(f"  efficiency   : {cru['eta']:8.3f}")
        print(f"  M_tip        : {cru['M_tip']:8.3f}")
        print(f"  max AoA      : {np.rad2deg(cru['alpha_max']):8.2f} deg")
        print(f"Maximum torque exerted on motor shaft: {cru['P']*60/((cru['n_rpm'])*2*np.pi):.1f} Nm")
        v = "OK" if spl <= NOISE_LIMIT_CR_DB else f"EXCEEDS by {spl-NOISE_LIMIT_CR_DB:.1f}"
        print(f"  cruise noise : {spl:8.1f} dB @ {NOISE_OBS_DIST_M:.0f} m "
              f"(limit {NOISE_LIMIT_CR_DB}: {v})")
        dba, _ = spl_to_dba(cru, blade, spl)
        print(f"  cruise noise : {dba:8.1f} dBA (A-weighted)")
 
    # ---------- PLOTS ----------
    print("\n--- Generating C_T plots ---")
    make_figure(blade)
 
    print("\n--- Selecting cruise pitch from efficiency diagram ---")
    sel = make_efficiency_figure(blade)
    if sel:
        print(f"  OPTIMUM cruise pitch:")
        print(f"    beta @ 0.75R : {sel['beta']:8.1f} deg")
        print(f"    collective   : {sel['collective']:8.1f} deg")
        print(f"    RPM          : {sel['rpm']:8.0f}")
        print(f"    efficiency   : {sel['eta']:8.3f}")
    else:
        print("  No feasible cruise pitch under current constraints.")
 
    print("\nNOTE: noise chart functions are DIGITIZED approximations (+/-10 dB).")
    pwl_c = pwl_regression(cru, blade, "cruise")
    spl_c_dir = pwl_to_spl_directional(pwl_c, NOISE_OBS_DIST_M, NOISE_OBS_THETA, hemisphere=True)
    print(f"  cruise SPL (Wang2025, theta={NOISE_OBS_THETA:.0f} deg): "f"{spl_c_dir:8.1f} dBA")
if __name__ == "__main__":
    main()

