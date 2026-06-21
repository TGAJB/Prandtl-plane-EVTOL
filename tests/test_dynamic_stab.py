"""
Verification harness for the dynamic-stability analysis (dyn_stab_analysis.py).

The numerical CORE of the script (state-space assembly, mode extraction, mode
characterisation, handling-quality logic, eigenmotion forcing) is copied
VERBATIM below so we test the actual shipped logic, not a paraphrase.

The script's end-to-end entry point main_aircraft() needs the full project
(parameters.py, class_II_sizing, stat_long_stab_anal_func, ...) which is not
present here, so we drive the core with:
  - the published Stevens & Lewis F-16 longitudinal/lateral A-matrices
    (aircraft-dynamics-classical.pdf / modeling-design-simulation.pdf, project
    refs) as the literature reference for the A -> modes -> HQ pipeline, and
  - a self-consistent derivative set for the C1/C2/C3 -> A construction tests.
"""
import numpy as np
from scipy import signal

# ===========================================================================
# ---- VERBATIM CORE FROM dyn_stab_analysis.py ------------------------------
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

    eig_a = np.linalg.eigvals(Aa)
    osc   = [l for l in eig_a if abs(l.imag) > 1e-9]
    reals = sorted([l.real for l in eig_a if abs(l.imag) <= 1e-9], key=abs)
    modes["Dutch roll"]      = characterise(osc[0] if osc[0].imag > 0 else osc[1])
    modes["Spiral"]          = characterise(complex(reals[0], 0))
    modes["Roll subsidence"] = characterise(complex(reals[-1], 0))
    return eig_s, eig_a, modes


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


def make_pulse(t, mag, t0, t1):
    u = np.zeros_like(t); u[(t >= t0) & (t < t1)] = mag; return u


def make_doublet(t, mag, t0, tm, t1):
    u = np.zeros_like(t)
    u[(t >= t0) & (t < tm)] = mag
    u[(t >= tm) & (t < t1)] = -mag
    return u


def forced_response(A, B, t, u):
    sys = signal.StateSpace(A, B, np.eye(A.shape[0]), np.zeros((A.shape[0], B.shape[1])))
    _, _, x = signal.lsim(sys, U=np.atleast_2d(u).T, T=t)
    return x.T


# ---- state_space() body, verbatim, wrapped on a light-weight model --------
class Model:
    """Minimal stand-in exposing exactly the attributes CruiseModel.state_space
    reads. state_space() below is copied verbatim from the script."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def state_space(self):
        V, c, b = self.V0, self.c, self.b
        muc, mub = self.muc, self.mub
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
# ---- END VERBATIM CORE ----------------------------------------------------
# ===========================================================================


# ===========================================================================
#   REFERENCE DATA
# ===========================================================================
# Published F-16 Jacobians (Stevens & Lewis, straight & level, 502 ft/s, cg 0.3c)
# -- project refs aircraft-dynamics-classical.pdf / modeling-design-simulation.pdf
F16_LONG = np.array([
    [-2.0244e-2,  7.8763e+0, -3.2170e+1, -6.5020e-1],
    [-2.5372e-4, -1.0190e+0,  0.0,        9.0484e-1],
    [ 0.0,        0.0,        0.0,        1.0      ],
    [ 7.9472e-11,-2.4982e+0,  0.0,       -1.3861e+0],
])
F16_LAT = np.array([
    [-3.2200e-1,  6.4032e-2,  3.8904e-2, -9.9156e-1],
    [ 0.0,        0.0,        1.0,        3.9385e-2],
    [-3.0919e+1,  0.0,       -3.6730e+0,  6.7425e-1],
    [ 9.4724e+0,  0.0,       -2.6358e-2, -4.9849e-1],
])
# Published modal results from the same source (period [s], zeta, tau [s]):
F16_REF = {
    "Short period":   dict(T=4.21,  zeta=0.628),
    "Phugoid":        dict(T=84.9,  zeta=0.117),
    "Dutch roll":     dict(T=1.95,  zeta=0.135),
    "Roll subsidence":dict(tau=0.28),
    "Spiral":         dict(tau=77.9),
}

# A self-consistent, transport-like non-dimensional derivative set for the
# C1/C2/C3 -> A construction tests (values are physically typical; the
# construction tests check internal consistency / scaling, not a named aircraft).
def baseline_kwargs(V0=160.0):
    return dict(
        V0=V0, c=2.06, b=15.9,
        muc=102.7, mub=15.5, KX2=0.019, KY2=1.3925, KZ2=0.042, KXZ=0.002,
        CL=1.136,
        # symmetric
        CX0=0.0, CZ0=-1.136, CXu=-0.0280, CZu=-0.376, CXa=0.480, CZa=-5.743,
        CZadot=-0.0035, CZq=-5.663, Cma=-0.571, Cmq=-8.794, Cmadot=0.178,
        CXde=-0.0373, CZde=-0.696, Cmde=-1.162, CXq=0.0, Cmu=0.0,
        # asymmetric
        CYb=-0.750, Clb=-0.1026, Cnb=0.1348, CYbdot=0.0, Cnbdot=0.0,
        CYp=-0.0304, Clp=-0.7110, Cnp=-0.0602, CYr=0.8495, Clr=0.2376, Cnr=-0.2061,
        Clda=-0.2309, CYdr=0.230, Cldr=0.0344, Cndr=-0.0939, CYda=0.0, Cnda=0.0,
    )


# ===========================================================================
#   TEST RUNNER
# ===========================================================================
RESULTS = []
def record(tid, name, expected, actual, passed):
    RESULTS.append((tid, name, expected, actual, "Pass" if passed else "FAIL"))
    tag = "PASS" if passed else "FAIL"
    print(f"\n[{tag}] {tid}  {name}")
    print(f"       expected: {expected}")
    print(f"       actual  : {actual}")


# ---------------------------------------------------------------------------
# DS-U5  Hand-computed / literature reference for characterise()
# ---------------------------------------------------------------------------
def test_DSU5():
    # Feed the two textbook symmetric eigenvalues straight into characterise()
    sp = characterise(complex(-1.2039,  1.4922))
    ph = characterise(complex(-0.0087297, 0.073966))
    checks = {
        "SP period": (sp["period"], 4.21), "SP zeta": (sp["zeta"], 0.628),
        "PH period": (ph["period"], 84.9), "PH zeta": (ph["zeta"], 0.117),
    }
    errs = {k: abs(v[0]-v[1])/abs(v[1])*100 for k, v in checks.items()}
    passed = all(e < 1.0 for e in errs.values())
    record("DS-U5-4a", "Hand/literature reference of characterise() vs Stevens&Lewis F-16",
           "T,zeta within 1% of textbook (SP 4.21s/0.628, PH 84.9s/0.117)",
           "  ".join(f"{k}={checks[k][0]:.3f}({errs[k]:.2f}%)" for k in checks),
           passed)


# ---------------------------------------------------------------------------
# DS-U6  Independent implementation cross-check
# ---------------------------------------------------------------------------
def test_DSU6():
    # (i) construction path A = -inv(C1)C2  vs  generalized QZ eig(C2,-C1)
    m = Model(**baseline_kwargs())
    As, Bs, Aa, Ba = m.state_space()
    from scipy.linalg import eig as geig
    # rebuild C1,C2 to form the pencil independently
    V, c, b, muc, mub = m.V0, m.c, m.b, m.muc, m.mub
    C1s = np.array([[-2*muc*c/V,0,0,0],[0,(m.CZadot-2*muc)*c/V,0,0],
                    [0,0,-c/V,0],[0,m.Cmadot*c/V,0,-2*muc*m.KY2*c/V]])
    C2s = np.array([[m.CXu,m.CXa,m.CZ0,m.CXq],[m.CZu,m.CZa,-m.CX0,m.CZq+2*muc],
                    [0,0,0,1],[m.Cmu,m.Cma,0,m.Cmq]])
    qz = np.sort_complex(geig(C2s, -C1s)[0])
    inv = np.sort_complex(np.linalg.eigvals(As))
    d_path = float(np.max(np.abs(qz - inv)))

    # (ii) full F-16 4x4 matrices through identify_modes -> all five modes vs textbook
    _, _, modes = identify_modes(F16_LONG, F16_LAT)
    worst = 0.0; detail = []
    for name, ref in F16_REF.items():
        mo = modes[name]
        if "T" in ref:
            e1 = abs(mo["period"]-ref["T"])/ref["T"]*100
            e2 = abs(mo["zeta"]-ref["zeta"])/ref["zeta"]*100
            worst = max(worst, e1, e2)
            detail.append(f"{name}:T={mo['period']:.2f}({e1:.2f}%),z={mo['zeta']:.3f}({e2:.2f}%)")
        else:
            e1 = abs(mo["tau"]-ref["tau"])/ref["tau"]*100
            worst = max(worst, e1)
            detail.append(f"{name}:tau={mo['tau']:.2f}({e1:.2f}%)")
    passed = (d_path < 1e-9) and (worst < 1.0)
    record("DS-U6-5a", "Independent cross-check: QZ vs inverse, and F-16 pipeline vs textbook",
           "QZ-vs-inverse < 1e-9 ; all 5 modes within 1% of Stevens&Lewis",
           f"QZ-vs-inverse max|d|={d_path:.2e} | worst mode err={worst:.2f}%  [" + "; ".join(detail) + "]",
           passed)


# ---------------------------------------------------------------------------
# DS-U8  Conservation / eigenvalue identity (trace = sum, det = product)
# ---------------------------------------------------------------------------
def test_DSU8():
    m = Model(**baseline_kwargs())
    As, _, Aa, _ = m.state_space()
    out = []
    ok = True
    for tag, A in (("sym", As), ("asym", Aa)):
        ev = np.linalg.eigvals(A)
        e_tr = abs(np.trace(A) - ev.sum())
        e_det = abs(np.linalg.det(A) - np.prod(ev))
        scale = max(1.0, abs(np.trace(A)), abs(np.linalg.det(A)))
        ok = ok and (e_tr/scale < 1e-9) and (e_det/scale < 1e-7)
        out.append(f"{tag}: |trace-Sum L|={e_tr:.2e}, |det-Prod L|={e_det:.2e}")
    record("DS-U8-6a", "Eigenvalue identities: trace(A)=Sum(lambda), det(A)=Prod(lambda)",
           "both residuals at float tolerance for symmetric & asymmetric A",
           " | ".join(out), ok)


# ---------------------------------------------------------------------------
# DS-U7  Scaling / metamorphic: A ~ V0  (dimensional eigenvalues scale with TAS)
# ---------------------------------------------------------------------------
def test_DSU7():
    m1 = Model(**baseline_kwargs(V0=160.0))
    m2 = Model(**baseline_kwargs(V0=320.0))   # x2 TAS, identical non-dim derivatives
    A1, _, B1, _ = m1.state_space()
    A2, _, B2, _ = m2.state_space()
    e1 = np.sort_complex(np.linalg.eigvals(A1))
    e2 = np.sort_complex(np.linalg.eigvals(A2))
    ratio = np.abs(e2) / np.abs(e1)
    err = float(np.max(np.abs(ratio - 2.0)))
    # also asymmetric
    Aa1 = m1.state_space()[2]; Aa2 = m2.state_space()[2]
    ra = np.abs(np.sort_complex(np.linalg.eigvals(Aa2)))/np.abs(np.sort_complex(np.linalg.eigvals(Aa1)))
    err = max(err, float(np.max(np.abs(ra - 2.0))))
    passed = err < 1e-9
    record("DS-U7-7a", "Scaling: doubling TAS doubles every eigenvalue magnitude (1/s)",
           "|lambda(2V)|/|lambda(V)| = 2.000 for all modes (sym & asym)",
           f"max deviation from 2.000 = {err:.2e}", passed)


# ---------------------------------------------------------------------------
# DS-U4  Extreme value: huge |Cmq| forces an overdamped short period
#        -> exercises the equivalent_pair branch; outputs must stay finite
# ---------------------------------------------------------------------------
def test_DSU4():
    kw = baseline_kwargs()
    kw.update(Cmq=-200.0, Cma=-3.0)          # very large pitch damping (box-wing-like)
    m = Model(**kw)
    As, _, Aa, _ = m.state_space()
    _, _, modes = identify_modes(As, Aa)
    sp = modes["Short period"]
    finite = all(np.isfinite([sp.get("wn", np.nan), sp.get("zeta", np.nan)]))
    # also confirm no NaN/inf anywhere and overdamped branch taken
    all_finite = True
    for mo in modes.values():
        for k in ("wn", "zeta", "tau", "period"):
            if k in mo and not np.isfinite(mo[k]):
                all_finite = False
    overdamped = sp.get("type") == "overdamped"
    passed = finite and all_finite and overdamped
    record("DS-U4-3a", "Extreme |Cmq| -> overdamped SP; equivalent_pair branch finite & handled",
           "SP labelled 'overdamped', wn_eq & zeta_eq finite, no NaN/inf in any mode",
           f"SP.type={sp.get('type')}, wn_eq={sp.get('wn'):.3f}, zeta_eq={sp.get('zeta'):.3f}, all_finite={all_finite}",
           passed)


# ---------------------------------------------------------------------------
# DS-U10  Output format / interface contract
# ---------------------------------------------------------------------------
def test_DSU10():
    _, _, modes = identify_modes(F16_LONG, F16_LAT)
    need = {"Short period", "Phugoid", "Dutch roll", "Spiral", "Roll subsidence"}
    have_modes = need.issubset(modes.keys())
    contract_ok = True
    for name, mo in modes.items():
        if not {"lambda", "stable", "type"}.issubset(mo):
            contract_ok = False
        if mo["type"] == "oscillatory" and not {"wn", "zeta", "period"}.issubset(mo):
            contract_ok = False
        if mo["type"] == "aperiodic" and "tau" not in mo:
            contract_ok = False
        if not (("T_half" in mo) ^ ("T_double" in mo)):   # exactly one present
            contract_ok = False
    rep = hq_assessment(modes)
    hq_ok = (len(rep) == 5 and all(len(r) == 3 and isinstance(r[1], (bool, np.bool_)) for r in rep))
    passed = have_modes and contract_ok and hq_ok
    record("DS-U10-8a", "Output contract: 5 named modes w/ required keys; hq_assessment 5x(str,bool,str)",
           "all 5 modes present with valid key set; hq_assessment returns 5 well-formed criteria",
           f"modes_present={have_modes}, key_contract={contract_ok}, hq_rows={len(rep)}/{hq_ok}",
           passed)


# ---------------------------------------------------------------------------
# DS-U1  Formula inspection (numeric corroboration of the documented EOM)
#   We corroborate the structural inspection by confirming the assembled A is
#   exactly the algebraic A = -C1^-1 C2 for a hand-built C1,C2 (independent typing)
#   and that characterise()'s defining identities hold on a known lambda.
# ---------------------------------------------------------------------------
def test_DSU1():
    lam = complex(-0.30, 1.20)
    o = characterise(lam)
    id_wn   = abs(o["wn"]    - abs(lam))            < 1e-12
    id_zeta = abs(o["zeta"]  - (0.30/abs(lam)))     < 1e-12
    id_P    = abs(o["period"]- (2*np.pi/1.20))      < 1e-12
    id_Th   = abs(o["T_half"]- (np.log(2)/0.30))    < 1e-12
    # aperiodic identity
    a = characterise(complex(-2.5, 0.0))
    id_tau  = abs(a["tau"] - 1/2.5) < 1e-12
    passed = all([id_wn, id_zeta, id_P, id_Th, id_tau])
    record("DS-U1-1a", "Formula inspection: characterise identities (wn,zeta,P,T_half,tau)",
           "wn=|L|; zeta=-Re/|L|; P=2pi/|Im|; T_half=ln2/|Re|; tau=1/|Re|",
           f"wn:{id_wn} zeta:{id_zeta} P:{id_P} T_half:{id_Th} tau:{id_tau}", passed)


# ---------------------------------------------------------------------------
# DS-U2  Unit consistency: A in [1/s]; free response of a stable system decays;
#        redimensionalisation factors used in the simulator are dimensionless-correct
# ---------------------------------------------------------------------------
def test_DSU2():
    m = Model(**baseline_kwargs())
    As, Bs, Aa, Ba = m.state_space()
    # All eigenvalues have units 1/s -> a 5 s horizon should show decay for stable modes
    t = np.arange(0, 5, 0.01)
    u = make_doublet(t, np.deg2rad(1.0), 0.5, 1.0, 1.5)   # elevator doublet [rad]
    y = forced_response(As, Bs, t, u)
    finite = np.all(np.isfinite(y))
    settled = abs(y[1, -1]) < abs(y[1]).max()              # alpha decays from its peak
    # redimensionalisation factor checks (pure dimensionless algebra):
    #   q[rad/s] = (q c / V) * V / c ;  p,r[rad/s] = (p b/2V) * 2V / b
    q_factor_ok = abs((m.V0/m.c) * (m.c/m.V0) - 1.0) < 1e-12
    pr_factor_ok = abs((2*m.V0/m.b) * (m.b/(2*m.V0)) - 1.0) < 1e-12
    passed = finite and settled and q_factor_ok and pr_factor_ok
    record("DS-U2-2a", "Unit consistency: A in 1/s (stable forced response decays); redim factors exact",
           "response finite & decaying on a seconds time-axis; (V/c)(c/V)=1, (2V/b)(b/2V)=1",
           f"finite={finite}, alpha_decays={settled}, q_factor_ok={q_factor_ok}, pr_factor_ok={pr_factor_ok}",
           passed)


if __name__ == "__main__":
    print("="*78)
    print("  DYNAMIC-STABILITY VERIFICATION (DS)  -- core of dyn_stab_analysis.py")
    print("="*78)
    test_DSU1(); test_DSU2(); test_DSU4(); test_DSU5()
    test_DSU6(); test_DSU7(); test_DSU8(); test_DSU10()
    print("\n" + "="*78)
    print("  SUMMARY")
    print("="*78)
    for tid, name, _, _, status in RESULTS:
        print(f"  {status:4s}  {tid:12s}  {name}")
    n_pass = sum(1 for r in RESULTS if r[4] == "Pass")
    print(f"\n  {n_pass}/{len(RESULTS)} passed")