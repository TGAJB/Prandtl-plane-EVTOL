# Final Design — Consolidated Report

## Active design (override)

| variable | value |
|---|---|
| `x_LEMAC_fw` | 0.169376 |
| `x_batt_cruise` | 1.53308 |
| `x_batt_vtol` | 6.400185 |
| `z_w_fw` | -0.858393 |
| `b_vert_tail` | 1.4 |
| `design_point` | 765.0 |
| `s_aft_to_s_total` | 0.6 |
| `t_nonfolding` | 0.005 |

## Mass & MTOW [kg]

| component | mass |
|---|---|
| mtow | 1979.91 |
| fuselage | 126.91 |
| wing | 198.32 |
| winglet | 22.58 |
| landing_gear | 41.44 |
| tail | 20.85 |
| motors | 193.20 |
| props | 70.25 |
| hubs | 41.72 |
| battery | 539.88 |
| payload | 400.00 |
| misc | 296.78 |
| hinge | 27.97 |
| max_power_kW | 930.8 |

## Wing geometry

| quantity | value |
|---|---|
| total_area_m2 | 25.6161 |
| aspect_ratio | 6.5974 |
| root_chord_m | 1.3589 |
| tip_chord_m | 0.6115 |
| mac_m | 1.0325 |
| span_m | 13.0000 |

## Inertia Cruise

- total mass: 1979.91 kg
- c.g. (x,y,z): [2.706938767033608, 7.177540971697626e-18, -0.2325546153465785]
- Ixx/Iyy/Izz: 9087.18145753138, 6642.01890988473, 13646.948535638308
- Ixz: 1936.7773033998897

## Inertia Vtol

- total mass: 1979.91 kg
- c.g. (x,y,z): [4.601657550008399, 7.177540971697626e-18, -0.2325546153465785]
- Ixx/Iyy/Izz: 9087.18145753138, 9074.144576363826, 16079.074202117405
- Ixz: 736.356930084004

## Stability

- MTOW used: 1979.905760376435
- accepted: **True**   worst normalised margin: -1.0

### Requirements

| requirement | pass | margin |
|---|---|---|
| C_M_alpha | PASS | -6.120689617008492 |
| C_L_q | PASS | -13.279044622625454 |
| C_M_q | PASS | -104.25104960473453 |
| C_Y_p | PASS | -0.008873205434661064 |
| C_L_p | PASS | -0.3863205737309413 |
| C_N_p | PASS | -0.04725227911957237 |
| C_Y_beta | PASS | -2.1807118670738554 |
| C_L_beta | PASS | -0.09810033225886164 |
| C_N_beta | PASS | -0.14396497159339172 |
| C_Y_r | PASS | -0.35712859035546063 |
| C_L_r | PASS | -0.13515206996709117 |
| C_N_r | PASS | -0.10878743736289033 |
| cg_cruise_fwd_ok | PASS | -2.267324670806213 |
| cg_cruise_aft_ok | PASS | -0.03037435824252066 |
| cg_vtol_fwd_ok | PASS | -0.10043466543331636 |
| cg_vtol_aft_ok | PASS | -0.317760357821256 |

### Stability derivatives

| derivative | value |
|---|---|
| C_L_alpha | 6.06731751073165 |
| C_M_alpha | -6.120689617008492 |
| C_L_alpha_dot | 4.296728063237054 |
| C_M_alpha_dot | -16.192412800232713 |
| C_L_q | 13.279044622625454 |
| C_M_q | -104.25104960473453 |
| C_Y_beta | -2.1807118670738554 |
| C_L_beta | -0.09810033225886164 |
| C_N_beta | 0.14396497159339172 |
| C_Y_beta_dot | 0.011409870201154287 |
| C_N_beta_dot | -0.0033007793756290324 |
| C_Y_p | -0.008873205434661064 |
| C_L_p | -0.3863205737309413 |
| C_N_p | -0.04725227911957237 |
| C_Y_r | 0.35712859035546063 |
| C_L_r | 0.13515206996709117 |
| C_N_r | -0.10878743736289033 |
| C_X_0 | -0.0 |
| C_Z_0 | -0.592991707858152 |
| C_X_u | -0.025060821358909347 |
| C_Z_u | -1.2037940361705124 |
| C_X_alpha | 0.33657672220273654 |
| C_Z_alpha | -6.079847921411105 |
| C_Z_alpha_dot | -4.296728063237054 |
| C_Z_q | -13.279044622625454 |
| C_L_delta_e | 0.7996507196827296 |
| C_M_delta_e | -3.013519673235035 |
| C_Z_delta_e | -0.7996507196827296 |
| C_X_delta_e | 0.0 |
| C_Y_delta_r | 0.14143661563144583 |
| C_N_delta_r | -0.040916421975406694 |
| C_L_delta_r | 0.009928356644113533 |
| C_L_delta_a | 0.03618434703729836 |
| C_N_delta_a | 0.0 |

## Matching diagram

- feasible W/S window: [459.4, 765.6] N/m²

## Department runs (graphs / console)

| department | status |
|---|---|
| matching_diagram | ok |
| aero_model | ok |
| drag_polar | ok |
| propeller_ppe | ok |
| ecs | ok |
| aux_loads | ok |
| static_long_stab | ok |
| vtol_oei | ok |
| dyn_stab | failed: module 'control' has no attribute 'ss' |
| landing_gear_trade | ok |

## Parameter list

Full parameter dump: `data/parameters_full.txt` / `.json` (234 module-level constants + the AircraftParameters dataclass tree).

## Figures

All graphs are consolidated in `figures/`. The Sobol/Tornado sensitivity heatmaps are copied in as standalone artifacts (NOT recomputed by this run).
