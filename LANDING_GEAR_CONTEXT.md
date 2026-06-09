# Landing-Gear Sizing — Context / Handoff

A self-contained briefing for a fresh assistant picking up the landing-gear work in this
repo. Read this top to bottom; it explains what exists, why, and the non-obvious gotchas.

---

## 1. Project at a glance

- Repo: `Prandtl-plane-EVTOL` — Class-II mass sizing of a Prandtl-plane (box-wing) eVTOL.
- Branch: `improvements_to_mass_components`.
- Language: Python. **Use `python3`** (the bare `python` command does not exist on this
  machine). Python 3.14; `numpy`, `scipy`, `matplotlib`, `pandas` are installed.
- Core pipeline: `class_II_sizing/mtow_sizing.py` runs a fixed-point loop that sums component
  masses (fuselage, wing, **landing gear**, tail, motors, props, battery, …) until MTOW
  converges.
- Hard rule for this codebase: **functions, not classes**. SI units. Constants carry a
  `# [unit] description` comment. Constants live in `parameters.py` (repo root).

---

## 2. What was done (recent work, in order)

1. **Dissolved `class_II_sizing/landing_gear.py`** (a standalone 6-architecture skid-gear
   trade study using scipy SLSQP + Monte-Carlo scoring + an F-delta plot) into the sizing
   pipeline. That file is **deleted**. Its pieces now live in three places:
   - physics + sizing + in-loop scoring → `mass_components.py`
   - all constants → `parameters.py`
   - offline reporting (scoring report, MC sensitivity, plot) → new
     `class_II_sizing/trade_off_landing_gear.py`
2. **Replaced** the previous landing-gear model in `mass_components.py` (an older
   4-architecture, sink-speed, numpy-grid model: `_size_plastic`/`_size_elastic`/
   `_size_leaf_spring`/`_governing_plastic_tube`/`_arch_*`/`_select_gear`) — all of that is
   gone.
3. **Removed the oleo-strut architecture** ("D oleo strut") everywhere — the team committed
   to a skid-type gear, so the oleo concept and its constants (`OLEO_*`) were deleted. Five
   architectures remain.
4. **Auto-scaled the SLSQP search bounds with MTOW** (`_scaled_gear_bounds`) — the original
   bounds were calibrated at 700 kg and starved the leaf concepts of feasibility at the real
   ~2200 kg, which made the elastomer trivially "win 100%". Bounds now grow with mass.
5. **Raised SLSQP multi-starts to 48** (`GEAR_OPT_RESTARTS`) for stable feasibility/optima.
6. **Refactored the trade-off scoring code** for readability (explicit loops, clear names) —
   verified bit-identical to the originals; no math changed.
7. Produced a LaTeX writeup of the method (delivered in chat, not committed).

---

## 3. Current file map

| File | Role |
|------|------|
| `parameters.py` (repo root) | All constants. Landing-gear section = "Landing-gear drop trade study (CS-27.725 limit + 27.727 reserve)". |
| `class_II_sizing/mass_components.py` | One function per component. Holds the landing-gear sizing + scoring (details below). |
| `class_II_sizing/mtow_sizing.py` | `compute_mtow()` + convergence loop. Calls `landing_gear_mass(mtow_kg)`. |
| `class_II_sizing/trade_off_landing_gear.py` | Offline study: `sensitivity()`, `plot()`, `main()`. `MTOW_STUDY = 2198.58`. Runnable. |
| `class_II_sizing/landing_gear_sensitivity.py` | Tornado / Sobol / Latin-Hypercube robustness study on the new model. Runnable. |
| `tests/test_landing_gear.py` | 8 pytest tests for the new model. |
| `class_II_sizing/landing_gear.py` | **DELETED** (dissolved into the three files above). |
| `OLDmain.py` | Legacy, **out of scope**; still calls the old `landing_gear_mass(...)` tuple signature and will error if run. Leave unless asked. |

---

## 4. How the landing-gear sizing works (in `mass_components.py`)

Skid gear = 2 longitudinal rails joined by 2 transverse cross-members, 2 knees each → 4
"legs". Every architecture is an **energy absorber** sized for the two CS-27 drops. All
quantities are whole-gear totals (per-member × count, see `whole_gear`).

**Data flow:** `landing_gear_mass(mtow_kg)` →
`effective_mass` (drop mass `m_eff`, = MTOW since `LIFT=0`) →
`size_all_architectures(m_eff)` →
for each arch: `_scaled_gear_bounds` then `size(...)` (scipy SLSQP, min mass s.t. 4
constraints, 48 restarts, seeded) →
`score(rows)` (weighted trade-off over feasible archs) →
return the **weighted-winner's** dict.

**Key functions (all functions, no classes):**
- `effective_mass(mtow_kg, h, d, L)` — CS-27.725(b) effective drop mass.
- `v_limit()`, `v_reserve()` — impact speeds from `H_L` (reserve = 1.5× height).
- `E_limit(delta, m_eff)`, `E_reserve(delta, m_eff)` — energy targets (KE + gravity over stroke).
- Concept models, each `(x, m)` → dict with `k, Ue, Up, Fmax, mass, dy, dmax, reusable`:
  - `cross_tube`  (A, hollow round tube, ductile, **not** reusable)
  - `metal_leaf`  (B, rectangular leaf, ductile, **reusable** at limit)
  - `composite_leaf` (B′, CFRP, brittle, not reusable)
  - `crushable`   (F, stiff leaf + honeycomb crush, not reusable)
  - `elastomeric` (E, block spring, hysteretic, reusable)
- `whole_gear(...)` — scales one member to the full gear (× count, + skid rails).
- `constraints / feasible / mass_of` — the 4 drop constraints (g1 elastic@limit, g2 survive
  reserve, g3 peak-decel ≤ `N_LIMIT`·W, g4 stroke ≤ `ENVELOPE`).
- `size(name, fn, mat, x0, bounds, varnames, m_eff)` — SLSQP min-mass with restarts; adds
  `feasible, params, SEA, MSe, MSr, npk`.
- `metrics / normalise / score_table / score` — the weighted trade-off (min-max normalise
  each of 7 criteria, weighted sum; `WEIGHTS`/`QUAL` from parameters).
- `_GEAR_CONCEPTS` — name → concept function map.
- `_scaled_gear_bounds(m_eff)` — grows each var's upper bound (and seed) by
  `max(1, (m_eff/GEAR_BOUNDS_REF_MASS)**scale_i)`.
- `size_all_architectures(m_eff)` — sizes all 5; used by the loop, the trade study, and the
  sensitivity script.
- `landing_gear_mass(mtow_kg, return_details=True)` — public entry; returns a details dict
  (`m_gear, arch, params, score, all_architectures, …`) or `None` if nothing is feasible
  (loop then falls back to 3% MTOW).

**Architecture lettering** (kept from the original study): A=cross_tube, B=metal_leaf,
B′=composite_leaf, F=crushable, E=elastomeric. (D=oleo was removed.)

---

## 5. Key constants (`parameters.py`, landing-gear section)

- Drop inputs: `H_L=0.25 m`, `D_EST=0.15 m`, `LIFT=0.0`, `N_LIMIT=20.0 g`, `ENVELOPE=0.40 m`.
- Materials (dicts, deliberately not scalars, to keep `eu`/`Gc`): `AL`, `STEEL`, `CFRP` =
  `{"E","sy","rho","eu","Gc"}`.
- Architecture counts: `N_SKID=2`, `SKID_RAIL_MASS=15`, `CROSSTUBE_COUNT=2`,
  `HINGES_PER_TUBE=2`, `LEAF_COUNT=2`, `HINGES_PER_LEAF=2`, `COMPOSITE_COUNT=2`,
  `CRUSH_COUNT=4`, `ELASTO_COUNT=4`.
- Structural factors: `BC_FACTOR=3.0`, `CROSS_SPAN=0.60`, `TUBE_HINGE_LEN=1.0`,
  `LEAF_HINGE_LEN=1.5`, `LEAF_DEV_FACTOR=2.0`, `COMP_CRUSH_FRAC=0.5`, `COMP_DELAM_FACTOR=2.0`,
  `CRUSH_LEAF_B=0.08`, `CRUSH_LEAF_L=0.45`, `HONEYCOMB_STRESS=2.5e6`, `HONEYCOMB_DENSITY=80`,
  `CRUSH_STROKE_EFF=0.75`, `ELASTO_FIXED_MASS=1.2`, `ELASTO_MASS_PER_N=1.0e-4`.
- Scoring: `QUAL` (per-arch tunable/cert_risk/cost) and `WEIGHTS` (criterion →
  `(direction, base_weight, uncertainty)`): SEA(max,0.25), mass(min,0.25), npk(min,0.15),
  reusable(max,0.10), tunable(max,0.08), cert_risk(min,0.12), cost(min,0.05).
- Optimizer: `GEAR_OPT_SEED=0`, `GEAR_OPT_RESTARTS=48`, `GEAR_BOUNDS_REF_MASS=700.0`,
  `GEAR_OPT_BOUNDS` (per-arch `x0`/`bounds`/`varnames`/`material`/`scale`).

`mass_components.py` imports these via `from parameters import (...)`.

---

## 6. Decisions & rationale (don't undo these without asking)

- **Drop-based scipy-SLSQP model chosen over the old sink-speed numpy-grid model** (user
  asked to integrate `landing_gear.py` and replace).
- **Oleo strut removed** — skid-type gear was selected.
- **Weighted scoring runs inside the MTOW loop** (user's explicit choice). The
  Monte-Carlo sensitivity stays offline in `trade_off_landing_gear.py`.
- **Bounds auto-scale with MTOW** (user chose this over a static retune).
- **Reporting / scoring left as-is** (the lone-survivor "100%" only appeared because of the
  stale bounds, which are now fixed).
- **Slow convergence accepted** — see gotchas.

---

## 7. Gotchas / known behavior (important)

- **~123-iteration MTOW convergence is BY DESIGN, not a bug.** With real competition the
  weighted score near-ties the heavy steel metal leaf (~285 kg) against the elastomer
  (~45 kg), so the in-loop winner flips as MTOW nudges until the mass settles. It converges
  (elastomer, ~2199 kg). The user accepted this; **do not** "fix" it by switching the loop to
  lightest-feasible or adding hysteresis unless asked.
- **Cross-tube is essentially always elastic-infeasible (g1)** — a bending tube cannot stay
  elastic at these sink speeds. That is correct physics, not a bounds problem.
- **Composite leaf is stroke-limited** at high mass (CFRP's low modulus → deflects past the
  0.40 m envelope). Real, not a bug.
- **SLSQP feasibility of the metal leaf is marginal** (small feasible island near the 20 g
  cap); 48 restarts make it stable but it can still flicker at isolated masses.
- `trade_off_landing_gear.py` writes `fd_curves.png` to the **current working directory**
  (relative path); it's a transient artifact, safe to delete.
- `landing_gear_sensitivity.py` overrides `mass_components` module globals at runtime
  (`_apply`/`_restore`) and runs the full sizer per sample (~60 s); writes PNGs + CSV to
  `class_II_sizing/sensitivity_plots/`.
- The `size()` docstring still says "8 random restarts" but the code uses
  `GEAR_OPT_RESTARTS=48`.

---

## 8. How to run / verify

```bash
cd "/Users/berke/Desktop/eVTOL repo/Prandtl-plane-EVTOL"
python3 -m pytest tests/test_landing_gear.py -q        # 8 tests, must pass
python3 class_II_sizing/mtow_sizing.py                 # full MTOW loop; converges ~2199 kg
python3 class_II_sizing/trade_off_landing_gear.py      # sizing+score+sensitivity tables, F-delta plot
python3 class_II_sizing/landing_gear_sensitivity.py    # tornado/Sobol/LHS (~60 s), writes plots+CSV
```

Sanity check the sizer directly:
```python
import class_II_sizing.mass_components as mc
d = mc.landing_gear_mass(2198.58)
print(d["arch"], round(d["m_gear"], 1), round(d["score"], 3))
# -> "E elastomeric" ~44.5 kg, score ~0.80
```

---

## 9. Persistent memory notes (in this Claude project)

- `landing-gear-elastic-infeasible` — a bending tube can't be an elastic spring.
- `landing-gear-bounds-autoscale` — bounds auto-scale with MTOW; the ~123-iter convergence
  from the near-tied weighted score is by-design; don't "fix" it.

---

## 10. Conventions to keep

- Functions only (no classes); SI units; `# [unit] description` comments on constants.
- New constants go in `parameters.py`; `mass_components.py` imports them by name.
- Keep equations/methods intact when refactoring — verify numerical equivalence (the scoring
  refactor was diffed bit-for-bit against the originals on the same inputs and RNG seed).
- Markdown file refs in chat use `[text](path)` links, not backticks.
