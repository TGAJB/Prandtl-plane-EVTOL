"""
dynamic_stability.py — Dynamic-stability analysis of the Folding Prandtl in CRUISE.

Builds the linearised symmetric and asymmetric equations of motion from the
DATCOM-derived stability & control derivatives (aircraft.solve()), extracts the
five classical eigenmotions, and checks them against MIL-F-8785C Level-1
handling-quality criteria.

State-space form, states and non-dimensionalisation follow the standard
TU Delft flight-dynamics formulation (same as the Citation 550 template):

    Symmetric  : x = [u_hat, alpha, theta, q*c/V]      input: delta_e
    Asymmetric : x = [beta,  phi,   p*b/2V, r*b/2V]    inputs: delta_a, delta_r

    C1 * x_dot + C2 * x + C3 * u = 0      ->     A = -C1^-1 C2,  B = -C1^-1 C3

Run:  python dynamic_stability.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np

try:
    import control as ct
    HAVE_CONTROL = True
except ImportError:                      # fall back to scipy if control is absent
    from scipy import signal
    HAVE_CONTROL = False

from parameters import AircraftParameters
from final_characteristics.vehicle_dynamics.aircraft import (
    Aircraft,
    Physical,
    FlightCondition,
    DatcomChartInputs,
    main_aircraft,
)


# ===========================================================================
# 1. INERTIA ESTIMATES (placeholders until the structures department delivers)
# ===========================================================================
# Roskam class-I radii-of-gyration method:  I = m * (Rbar * L_char / 2)^2
# Rbar values typical of light twin / commuter configurations. REPLACE the
# moments of inertia in MassProperties as soon as real numbers exist — every
# frequency below scales with 1/sqrt(I).
RBAR_X, RBAR_Y, RBAR_Z = 0.25, 0.35, 0.40


def fill_inertia_placeholders(params):
    m = params.mass
    b = params.wing_geometry.b_ref
    L = params.fuselage_geometry.fuselage_length
    e = 0.5 * (b + L)                                  # I_zz characteristic length
    filled = []
    if m.I_xx is None:
        m.I_xx = m.mtow * (RBAR_X * b / 2) ** 2;  filled.append("I_xx")
    if m.I_yy is None:
        m.I_yy = m.mtow * (RBAR_Y * L / 2) ** 2;  filled.append("I_yy")
    if m.I_zz is None:
        m.I_zz = m.mtow * (RBAR_Z * e / 2) ** 2;  filled.append("I_zz")
    if m.I_xz is None:
        m.I_xz = 0.05 * m.I_xx;                   filled.append("I_xz")
    if filled:
        print(f"[WARN] Inertias estimated by radius-of-gyration placeholders: {filled}")
        print(f"       I_xx={m.I_xx:.0f}  I_yy={m.I_yy:.0f}  I_zz={m.I_zz:.0f}  "
              f"I_xz={m.I_xz:.0f}  [kg m^2]")
    return params


# ===========================================================================
# 2. DERIVATIVE SHEET -> EOM COEFFICIENT SET
# ===========================================================================
class CruiseModel:
    """Collects everything the EOM need from one solved Aircraft instance."""

    # Derivatives the DATCOM sheet does not (yet) provide. All are small for a
    # clean subsonic propeller configuration; revisit when AVL data arrives.
    CXq    = 0.0      # X-force due to pitch rate            (template: -0.28)
    CXadot = 0.0      # X-force due to AoA rate              (template: +0.08)
    Cmu    = 0.0      # tuck derivative, ~0 at M = 0.17      (template: +0.07)
    CYda   = 0.0      # side force due to aileron
    Cnda   = 0.0      # ADVERSE YAW — placeholder! matters for Dutch-roll/
                      # aileron coordination; estimate ~ -k*CL*Clda, k=0.1-0.3

    def __init__(self, aircraft: Aircraft, solved):
        self.ac = aircraft
        p = solved                  # populates p.stability / p.controls
        s, c = p.stability, p.controls
        fc = aircraft.fc

        # flight condition
        self.V0  = fc.tas
        S_ref, b_ref, c_ref = aircraft._ref()
        self.S, self.b, self.c = S_ref, b_ref, c_ref
        self.CL = aircraft.CL_trim()            # gravity term in the asymmetric EOM

        # mass / inertia (dimensionless, template convention)
        self.muc, self.mub = aircraft.mu_c(), aircraft.mu_b()
        self.KX2, self.KY2 = aircraft.KX2(), aircraft.KY2()
        self.KZ2, self.KXZ = aircraft.KZ2(), aircraft.KXZ()

        # symmetric derivatives
        self.CX0, self.CZ0   = s.C_X_0, s.C_Z_0
        self.CXu, self.CZu   = s.C_X_u, s.C_Z_u
        self.CXa, self.CZa   = s.C_X_alpha, s.C_Z_alpha
        self.CZadot          = s.C_Z_alpha_dot
        self.CZq             = s.C_Z_q
        self.Cma, self.Cmq   = s.C_M_alpha, s.C_M_q
        self.Cmadot          = s.C_M_alpha_dot
        self.CXde, self.CZde = c.C_X_delta_e, c.C_Z_delta_e
        self.Cmde            = c.C_M_delta_e

        # asymmetric derivatives
        self.CYb, self.Clb, self.Cnb = s.C_Y_beta, s.C_L_beta, s.C_N_beta
        self.CYbdot, self.Cnbdot     = s.C_Y_beta_dot, s.C_N_beta_dot
        self.CYp, self.Clp, self.Cnp = s.C_Y_p, s.C_L_p, s.C_N_p
        self.CYr, self.Clr, self.Cnr = s.C_Y_r, s.C_L_r, s.C_N_r
        self.Clda                    = c.C_L_delta_a
        self.CYdr, self.Cldr, self.Cndr = c.C_Y_delta_r, c.C_L_delta_r, c.C_N_delta_r

    # -----------------------------------------------------------------------
    def state_space(self):
        """Return (As, Bs, Aa, Ba) — same C1/C2/C3 construction as the template."""
        V, c, b = self.V0, self.c, self.b
        muc, mub = self.muc, self.mub

        # ---- symmetric ----------------------------------------------------
        C1s = np.array([
            [-2*muc*c/V, 0,                          0,      0],
            [0,          (self.CZadot - 2*muc)*c/V,  0,      0],
            [0,          0,                          -c/V,   0],
            [0,          self.Cmadot*c/V,            0,      -2*muc*self.KY2*c/V],
        ])
        C2s = np.array([
            [self.CXu, self.CXa, self.CZ0,  self.CXq],
            [self.CZu, self.CZa, -self.CX0, self.CZq + 2*muc],
            [0,        0,        0,         1],
            [self.Cmu, self.Cma, 0,         self.Cmq],
        ])
        C3s = np.array([[-self.CXde], [-self.CZde], [0], [-self.Cmde]])

        As = -np.linalg.inv(C1s) @ C2s
        Bs = -np.linalg.inv(C1s) @ C3s

        # ---- asymmetric ---------------------------------------------------
        C1a = np.array([
            [(self.CYbdot - 2*mub)*b/V, 0,         0,                     0],
            [0,                         -b/(2*V),  0,                     0],
            [0,                         0,         -4*mub*self.KX2*b/V,   4*mub*self.KXZ*b/V],
            [self.Cnbdot*b/V,           0,         4*mub*self.KXZ*b/V,    -4*mub*self.KZ2*b/V],
        ])
        C2a = np.array([
            [self.CYb, self.CL, self.CYp, self.CYr - 4*mub],
            [0,        0,       1,        0],
            [self.Clb, 0,       self.Clp, self.Clr],
            [self.Cnb, 0,       self.Cnp, self.Cnr],
        ])
        C3a = np.array([
            [-self.CYda, -self.CYdr],
            [0,          0],
            [-self.Clda, -self.Cldr],
            [-self.Cnda, -self.Cndr],
        ])

        Aa = -np.linalg.inv(C1a) @ C2a
        Ba = -np.linalg.inv(C1a) @ C3a
        return As, Bs, Aa, Ba


# ===========================================================================
# 3. MODE EXTRACTION
# ===========================================================================
def characterise(lam):
    """Frequency / damping / time metrics of one eigenvalue (or complex pair)."""
    out = {"lambda": lam, "stable": lam.real < 0}
    if abs(lam.imag) > 1e-9:                                   # oscillatory
        wn = abs(lam)
        out.update(type="oscillatory", wn=wn, zeta=-lam.real/wn,
                   period=2*np.pi/abs(lam.imag),
                   T_half=np.log(2)/abs(lam.real) if lam.real != 0 else np.inf)
    else:                                                      # aperiodic
        out.update(type="aperiodic", tau=1/abs(lam.real) if lam.real != 0 else np.inf,
                   T_half=np.log(2)/abs(lam.real) if lam.real != 0 else np.inf)
    if not out["stable"]:
        out["T_double"] = out.pop("T_half")
    return out


def equivalent_pair(l1, l2):
    """Equivalent 2nd-order (wn, zeta) of two real roots (overdamped mode)."""
    wn = np.sqrt(l1*l2)
    return {"lambda": complex(l1, 0), "lambda2": complex(l2, 0),
            "stable": l1 < 0 and l2 < 0, "type": "overdamped",
            "wn": wn, "zeta": -(l1 + l2)/(2*wn),
            "T_half": np.log(2)/abs(max(l1, l2))}


def identify_modes(As, Aa):
    """Label the classical eigenmotions from the two A-matrices."""
    modes = {}

    # symmetric: short period & phugoid. Either mode may be overdamped (two
    # real roots) — common here because the box wing gives a very large |Cmq|.
    eig_s = np.linalg.eigvals(As)
    pairs = sorted({(round(l.real, 10), round(abs(l.imag), 10))
                    for l in eig_s if abs(l.imag) > 1e-9},
                   key=lambda rl: -np.hypot(*rl))
    reals_s = sorted([l.real for l in eig_s if abs(l.imag) <= 1e-9], key=abs)
    if len(pairs) == 2:                                   # both oscillatory
        modes["Short period"] = characterise(complex(*pairs[0]))
        modes["Phugoid"]      = characterise(complex(*pairs[1]))
    elif len(pairs) == 1 and len(reals_s) == 2:           # one mode overdamped
        pair_wn = np.hypot(*pairs[0])
        if pair_wn < np.sqrt(abs(reals_s[0]*reals_s[1])): # pair is the slow mode
            modes["Short period"] = equivalent_pair(*reals_s)
            modes["Phugoid"]      = characterise(complex(*pairs[0]))
        else:
            modes["Short period"] = characterise(complex(*pairs[0]))
            modes["Phugoid"]      = equivalent_pair(*reals_s)
    else:                                                 # all four real
        modes["Short period"] = equivalent_pair(reals_s[-1], reals_s[-2])
        modes["Phugoid"]      = equivalent_pair(reals_s[0], reals_s[1])

    # asymmetric: complex pair = Dutch roll; fastest real = roll subsidence;
    # remaining real = spiral
    eig_a = np.linalg.eigvals(Aa)
    osc   = [l for l in eig_a if abs(l.imag) > 1e-9]
    reals = sorted([l.real for l in eig_a if abs(l.imag) <= 1e-9], key=abs)
    modes["Dutch roll"]      = characterise(osc[0] if osc[0].imag > 0 else osc[1])
    modes["Spiral"]          = characterise(complex(reals[0], 0))
    modes["Roll subsidence"] = characterise(complex(reals[-1], 0))
    return eig_s, eig_a, modes


# ===========================================================================
# 4. HANDLING-QUALITY CHECKS  (MIL-F-8785C, Category B cruise, Level 1)
# ===========================================================================
def hq_assessment(modes):
    rep = []
    sp = modes["Short period"]
    rep.append(("Short period damping 0.30 <= zeta <= 2.00",
                0.30 <= sp.get("zeta", -1) <= 2.00, f"zeta = {sp.get('zeta', float('nan')):.3f}"))
    ph = modes["Phugoid"]
    rep.append(("Phugoid damping zeta >= 0.04",
                ph.get("zeta", -1) >= 0.04, f"zeta = {ph.get('zeta', float('nan')):.3f}"))
    dr = modes["Dutch roll"]
    ok = (dr.get("zeta", -1) >= 0.08 and dr.get("wn", 0) >= 0.4
          and dr.get("zeta", 0)*dr.get("wn", 0) >= 0.15)
    rep.append(("Dutch roll zeta>=0.08, wn>=0.4 rad/s, zeta*wn>=0.15",
                ok, f"zeta = {dr.get('zeta', float('nan')):.3f}, wn = {dr.get('wn', float('nan')):.2f}"))
    rs = modes["Roll subsidence"]
    rep.append(("Roll-mode time constant tau <= 1.4 s",
                rs["stable"] and rs.get("tau", 99) <= 1.4, f"tau = {rs.get('tau', float('nan')):.2f} s"))
    spm = modes["Spiral"]
    if spm["stable"]:
        rep.append(("Spiral: stable (or T_double >= 20 s)", True, "stable"))
    else:
        rep.append(("Spiral: T_double >= 20 s",
                    spm.get("T_double", 0) >= 20, f"T2 = {spm.get('T_double', float('nan')):.1f} s"))
    return rep


# ===========================================================================
# 5. EIGENMOTION SIMULATIONS (synthetic inputs — verification, as in template)
# ===========================================================================
def make_pulse(t, mag, t0, t1):
    u = np.zeros_like(t); u[(t >= t0) & (t < t1)] = mag; return u


def make_doublet(t, mag, t0, tm, t1):
    u = np.zeros_like(t)
    u[(t >= t0) & (t < tm)] = mag
    u[(t >= tm) & (t < t1)] = -mag
    return u


def forced_response(A, B, t, u):
    if HAVE_CONTROL:
        sys = ct.ss(A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1])))
        _, y = ct.forced_response(sys, T=t, U=u)
        return y
    sys = signal.StateSpace(A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1])))
    _, _, x = signal.lsim(sys, U=np.atleast_2d(u).T, T=t)
    return x.T


def simulate_eigenmotions(model, As, Bs, Aa, Ba, save_dir="./final_characteristics/vehicle_dynamics/plots/dyn_stab"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V, b, c = model.V0, model.b, model.c
    cfg = [
        ("Short period", "sym",  np.arange(0, 15, 0.01),
         lambda t: make_doublet(t, np.deg2rad(2.0), 1.0, 1.5, 2.0)),
        ("Phugoid",      "sym",  np.arange(0, 240, 0.05),
         lambda t: make_pulse(t, np.deg2rad(-2.0), 1.0, 4.0)),
        ("Dutch roll",   "asym", np.arange(0, 30, 0.01),
         lambda t: np.vstack((np.zeros_like(t), make_pulse(t, np.deg2rad(5.0), 1.0, 2.0)))),
        ("Aperiodic roll", "asym", np.arange(0, 10, 0.01),
         lambda t: np.vstack((make_pulse(t, np.deg2rad(-5.0), 0.5, 1.0), np.zeros_like(t)))),
        ("Spiral",       "asym", np.arange(0, 200, 0.05),
         lambda t: np.vstack((make_pulse(t, np.deg2rad(-1.0), 1.0, 3.0), np.zeros_like(t)))),
    ]
    files = []
    for name, kind, t, get_u in cfg:
        u = get_u(t)
        if kind == "sym":
            y = forced_response(As, Bs, t, u)
            data = [np.rad2deg(y[1]), np.rad2deg(y[2]), np.rad2deg(y[3]*V/c)]
            labels = ["alpha [deg]", "theta [deg]", "q [deg/s]"]
        else:
            y = forced_response(Aa, Ba, t, u)
            data = [np.rad2deg(y[0]), np.rad2deg(y[1]), np.rad2deg(y[2]*2*V/b),
                    np.rad2deg(y[3]*2*V/b)]
            labels = ["beta [deg]", "phi [deg]", "p [deg/s]", "r [deg/s]"]
        fig, axes = plt.subplots(len(data), 1, figsize=(10, 2.2*len(data)), sharex=True)
        fig.suptitle(f"{name} — cruise eigenmotion (synthetic input)", fontweight="bold")
        for ax, d, lab in zip(axes, data, labels):
            ax.plot(t, d, lw=1.1)
            ax.set_ylabel(lab); ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        f = f"{save_dir}/eigenmotion_{name.lower().replace(' ', '_')}.png"
        fig.savefig(f, dpi=130); plt.close(fig)
        files.append(f)
    return files


# ===========================================================================
# MAIN
# ===========================================================================
def main(plot=True, save_dir="./final_characteristics/vehicle_dynamics/plots/dyn_stab"):
    solved, aircraft = main_aircraft()

    model = CruiseModel(aircraft, solved)

    As, Bs, Aa, Ba = model.state_space()
    eig_s, eig_a, modes = identify_modes(As, Aa)

    print("\n" + "="*78)
    print("  FOLDING PRANDTL — CRUISE DYNAMIC STABILITY"
          f"   (V = {model.V0:.1f} m/s, CL = {model.CL:.3f})")
    print("="*78)
    print(f"  mu_c = {model.muc:.1f}   mu_b = {model.mub:.2f}   "
          f"KX2 = {model.KX2:.4f}  KY2 = {model.KY2:.3f}  KZ2 = {model.KZ2:.4f}")
    print(f"\n  Symmetric eigenvalues : {np.round(eig_s, 4)}")
    print(f"  Asymmetric eigenvalues: {np.round(eig_a, 4)}")

    print("\n  %-16s %-24s %-8s %s" % ("MODE", "lambda [1/s]", "stable", "characteristics"))
    print("  " + "-"*74)
    for name, m in modes.items():
        lam = m["lambda"]
        if m["type"] == "overdamped":
            lam_str = f"{lam.real:+.4f}, {m['lambda2'].real:+.4f}"
        elif m["type"] == "oscillatory":
            lam_str = f"{lam.real:+.4f} ± {abs(lam.imag):.4f}j"
        else:
            lam_str = f"{lam.real:+.5f}"
        if m["type"] == "overdamped":
            ch = (f"OVERDAMPED: wn_eq = {m['wn']:.3f} rad/s, zeta_eq = {m['zeta']:.3f}, "
                  f"T1/2 = {m['T_half']:.1f} s")
        elif m["type"] == "oscillatory":
            ch = (f"wn = {m['wn']:.3f} rad/s, zeta = {m['zeta']:.3f}, "
                  f"P = {m['period']:.1f} s, "
                  + (f"T1/2 = {m['T_half']:.1f} s" if m["stable"] else f"T2 = {m['T_double']:.1f} s"))
        else:
            ch = (f"tau = {m['tau']:.2f} s, "
                  + (f"T1/2 = {m['T_half']:.1f} s" if m["stable"] else f"T2 = {m['T_double']:.1f} s"))
        print("  %-16s %-24s %-8s %s" % (name, lam_str, "yes" if m["stable"] else "NO", ch))

    print("\n  Handling qualities (MIL-F-8785C, Class II/III Cat B, Level 1):")
    for crit, ok, detail in hq_assessment(modes):
        print(f"   [{'PASS' if ok else 'FAIL'}] {crit:<55s} {detail}")
    print("="*78 + "\n")

    if plot:
        files = simulate_eigenmotions(model, As, Bs, Aa, Ba, save_dir)
        print("Eigenmotion response plots written:")
        for f in files:
            print("  ", f)
    return model, As, Bs, Aa, Ba, modes


if __name__ == "__main__":
    main(plot=True)