# Root design variables — the tunable origin inputs for the stability optimiser

This is the reference list of **root design variables**: the bare, non-
interdependent *origin inputs* the team chooses, which then cascade (in the
sizing converger and the vehicle-dynamics solver) into the converged MTOW, the
stability & control derivatives, and the c.g. envelopes. Use it to decide which
`parameters.py` values are worth promoting into the optimiser search space
(`final_characteristics/optimiser.py` → `DESIGN_VARIABLES`).

**What counts as a root (and what does not).** A root is *set*, not *computed*.
Anything a method derives from other fields is excluded — it moves on its own
when its roots move. In particular the following are **NOT roots** and must never
be put in the search space directly:

- **c.g.** (`mass.x_cg_opt`, `x_cg`): an *emergent* output of the component
  mass/position build-up (MMOI). It is governed by the layout roots below
  (wing/tail/component positions) and the component sizing. In the sensitivity
  tool the c.g. is held at its design value and the c.g.-*envelope* limits move
  with the geometry roots. (The optimiser **currently still carries `x_cg` as a
  convenience *proxy* design variable** for ballast/placement; the principled
  follow-up is to drop it and let the layout roots above — `x_LEMAC_fw`,
  `x_vert_tail`, component positions — move the c.g. instead, which additionally
  needs the c.g. to be re-derived from those positions via MMOI in
  `evaluate_stability`.)
- **Derived geometry**: tail `S/AR/MAC/taper`, winglet `AR`, `z_w_aw`, `b_aw`,
  `taper_aw`, wing areas computed from `W/S`, all `MAC`s. Recomputed in
  `stability_eval._resolve_dependents` / `_update_aircraft_for_wing_split`.
- **Aero-department results**: `CL_alpha_*`, downwash, `x_ac_*`,
  `dyn_pres_ratio_*`, Oswald `e`, `C_M_ac_*`, `I_v`, section `cl_alpha`. These
  are inputs *to* this analysis produced by another method, not design knobs.

All line numbers below refer to root [`parameters.py`](../parameters.py).
**MTOW-coupled** roots feed the sizing converger through a
`class_II_sizing.mass_components` global (the name in the column); the others are
MTOW-neutral (they change geometry/derivatives only). The **leverage** column is
the *a-priori* expectation — the measured ranking is in the regenerated
`sensitivity_plots/stability/sobol_heatmap.png` + `sobol_total_order.csv`.

---

## A. Wing — planform & loading

| Root | loc | MTOW-coupled | Primarily drives | Leverage |
|---|---|---|---|---|
| `wing_geometry.design_point` (W/S) | :59 | **yes** `WING_LOADING_N` | `S_tot` → every area ratio → `C_M_alpha`, `C_L_q`/`C_M_q`, lateral derivs; `MAC`→cg envelope; wing/battery mass | **high** (MTOW **and** stability) |
| `s_aft_to_s_total` (area split) | pseudo-key | no¹ | front/aft balance → `C_M_alpha`, cg aft limit, `C_Y_beta`/`C_N_beta`, all area ratios | **high** (longitudinal) |
| `wing_geometry.taper_fw` | :79 | **yes** `TAPER_W` | sweep_c4 / AR effects → `C_L_beta`, `C_N_p`, `C_L_r`; `MAC`→cg; spar mass | med |
| `wing_geometry.b_fw` (span) | :67 | **yes** `WING_SPAN` | AR → `C_L_p`, `C_L_beta`, `C_N_p`, `C_L_r`; wing structural mass | **high** — *but see²* |

¹ `s_aft_to_s_total` is left **MTOW-neutral** in the sweep on purpose: the
converger global `AREA_SPLIT` is the *per-wing* area fraction used for structural
mass ([`mass_components.wing_geometry`](../class_II_sizing/mass_components.py)),
not the aft/total split, so the mapping is not 1:1. Its mass effect is second-
order; resolve the exact relation before coupling it.

² `b_fw` is a high-leverage root but is **not** in the tool's default sweep: the
box-wing reference fields it feeds (`b_ref`, and the reference area `S_ref` used
to non-dimensionalise the lateral derivatives) are not yet propagated on a span
override. `b_aw`/`taper_aw` already mirror via `_resolve_dependents`. Add the
`b_ref`/`S_ref` propagation before promoting span.

## B. Wing — box layout & 3-D shaping

| Root | loc | MTOW-coupled | Primarily drives | Leverage |
|---|---|---|---|---|
| `wing_geometry.stagger` | :74 | no | aft-wing volume → neutral point / cg aft limit; downwash; winglet sweep | **high** (cg) |
| `wing_geometry.gap` | :73 | no | `z_w_aw` (=`z_w_fw`+gap), downwash, winglet LE sweep | med |
| `wing_geometry.x_LEMAC_fw` | :122 | no | cg cruise fwd & aft limits (longitudinal datum) | **high** (cg) |
| `wing_geometry.z_w_fw` (front-wing height) | :119 | no | `C_L_beta` (wing-height effect), `C_L_p`, `C_Y_p` | **high** (`C_L_beta`) |
| `wing_geometry.dihedral_front_wing` | :90 | no | `C_L_beta` (primary), `C_Y_beta`, `C_L_p`, `C_Y_p` | **high** |
| `wing_geometry.dihedral_aft_wing` | :91 | no | same family | **high** |
| `wing_geometry.LE_sweep_fw` | :87 | no | sweep_c4 → `C_L_beta`, `C_N_p`, `C_L_r`, `C_Y_p` | med |
| `wing_geometry.LE_sweep_aw` | :88 | no | same family | med |
| `wing_geometry.twist_fw` / `twist_aw` | :93 / :94 | no | spanwise lift distribution → `C_L_beta` | low-med (flagged NOT FINAL) |

## C. Vertical tail

| Root | loc | MTOW-coupled | Primarily drives | Leverage |
|---|---|---|---|---|
| `tail_geometry.b_vert_tail` (span) | :139 | no³ | `C_Y_beta`, `C_N_beta`, `C_Y_r`, `C_N_r` (weathercock / yaw damping) | **high** (directional) |
| `tail_geometry.x_vert_tail` (arm) | :136 | no | `C_N_beta`, `C_N_r`, `C_Y_r`, `C_N_p` (moment arm); cg | **high** (directional) |
| `tail_geometry.c_r_vert_tail` | :137 | no³ | `S_vert_tail` → all directional derivs | med |
| `tail_geometry.c_t_vert_tail` | :138 | no³ | `S_vert_tail`, tail taper | med |
| `tail_geometry.z_vert_tail` | :149 | no | roll/yaw coupling (vertical arm `z_p`) | low-med |

³ Tail size currently does not feed the converger (`S_TAIL` mass uses its own
sheet value). If tail mass should respond, wire `b/c_r/c_t_vert_tail` →
`S_TAIL`/`AR_T` before treating them as MTOW-coupled.

## D. Winglet (vertical joiner)

| Root | loc | MTOW-coupled | Primarily drives | Leverage |
|---|---|---|---|---|
| `winglet_geometry.S_winglet` | :166 | no | winglet contribution to `C_Y_beta`, `C_L_beta`, `C_N_beta` | med |
| `winglet_geometry.b_winglet` | :167 | no | `AR_winglet`; lateral contributions | low-med |
| `winglet_geometry.x_ac_winglet` / `z_ac_winglet` | :172 / :173 | no | winglet moment arms (roll/yaw) | low-med |

## E. Fuselage

| Root | loc | MTOW-coupled | Primarily drives | Leverage |
|---|---|---|---|---|
| `fuselage_geometry.fuselage_length` | :186 | **yes** `L_FUS` | fuselage mass (Raymer) → MTOW; cg | **high (MTOW)** — top MTOW driver in the sweep (±~25 kg) |
| `fuselage_geometry.d_fw` (width) | :187 | yes⁴ `FUSE_WIDTH` | `C_L_beta` body term, side-area / `C_Y_p`, `C_N_beta`; landing-gear track | med (aero); **MTOW ≈ 0** |

⁴ `FUSE_WIDTH` feeds the landing-gear *track* (→ gear mass in the trade study),
not the fuselage-mass regression; over [1.6, 2.4] m its measured MTOW effect is
~0. Tagged coupled because it does enter the converger, but treat it as
MTOW-neutral in practice.

---

## F. VTOL-OEI c.g. roots — the levers for the always-failing requirements

The two requirements `cg_vtol_fwd_ok` / `cg_vtol_aft_ok` are infeasible at the
current design (the aft-propeller-out case cannot be balanced at MTOW), so they
fail in every run with the ±1e6 penalty. **No wing/tail geometry knob fixes
this** — consistent with the project note that VTOL-OEI needs a *propulsion*
change, not geometry. The actual roots are:

- the **VTOL rotor positions** `Propulsion.x_vtol_i` / `y_vtol_i` / `z_vtol_i`, and
- `Propulsion.max_thrust_per_engine`.

These live in the LP inside
[`vehicle_dynamics/VTOL_cg_envelope_det.py`](vehicle_dynamics/VTOL_cg_envelope_det.py)
(`load_design_parameters`), **not** the main parameter sheet, so a plain sheet
override does not move them. Exposing them to the sweep/optimiser requires wiring
those LP inputs through `evaluate_stability` — out of scope for the current
sheet-override tooling and recorded here as the known next lever.

---

## What is in the sensitivity sweep today

`final_characteristics/stability_sensitivity.py` sweeps the **19 highest-leverage
roots** above (sections A–E, excluding the `b_fw` span caveat² and the
NOT-FINAL twists / lower-impact winglet a.c. positions). Four are MTOW-coupled
(`design_point`, `taper_fw`, `fuselage_length`, `d_fw`); each sample reconverges
the MTOW so the heatmap ranks both stability-drivers and MTOW-drivers in one run.
Toggle the remaining roots in by editing the tool's `PARAMS` list. Run:

```
python final_characteristics/stability_sensitivity.py            # full (N_SOBOL=16)
STAB_SENS_N_SOBOL=4 python final_characteristics/stability_sensitivity.py   # quick
```

and read `sensitivity_plots/stability/sobol_heatmap.png` (per-(root × output)
total-order indices, with an MTOW row) and `sobol_total_order.csv`.
