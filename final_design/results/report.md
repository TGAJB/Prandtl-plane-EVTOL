# Final Design — Consolidated Report

## Active design (override)

| variable | value |
|---|---|
| `x_LEMAC_fw` | 0.17583 |
| `x_batt_cruise` | 1.507148 |
| `x_batt_vtol` | 6.418074 |
| `z_w_fw` | -0.895019 |
| `b_vert_tail` | 1.4 |
| `design_point` | 765.0 |
| `s_aft_to_s_total` | 0.6 |
| `t_nonfolding` | 0.005 |

## Mass & MTOW [kg]

| component | mass |
|---|---|
| mtow | 1979.90 |
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
| total_area_m2 | 25.6160 |
| aspect_ratio | 6.5974 |
| root_chord_m | 1.3589 |
| tip_chord_m | 0.6115 |
| mac_m | 1.0325 |
| span_m | 13.0000 |

## Inertia Cruise

- total mass: 1979.90 kg
- c.g. (x,y,z): [2.6998671337633136, 7.177544353012179e-18, -0.24280391001024]
- Ixx/Iyy/Izz: 9075.984246561293, 6663.956779560079, 13680.068777838569
- Ixz: 1930.7631178633146

## Inertia Vtol

- total mass: 1979.90 kg
- c.g. (x,y,z): [4.606535635163275, 7.177544353012179e-18, -0.24280391001024]
- Ixx/Iyy/Izz: 9075.984246561293, 9097.815755295795, 16113.927753574286
- Ixz: 716.8179747640721

## Stability

- MTOW used: 1979.9048276501667
- accepted: **True**   worst normalised margin: -1.0

### Requirements

| requirement | pass | margin |
|---|---|---|
| C_M_alpha | PASS | -6.160299353371169 |
| C_L_q | PASS | -13.37616426890122 |
| C_M_q | PASS | -104.09487789356604 |
| C_Y_p | PASS | -0.008873205441599571 |
| C_L_p | PASS | -0.386320573730939 |
| C_N_p | PASS | -0.046901118967110245 |
| C_Y_beta | PASS | -2.1808877618642937 |
| C_L_beta | PASS | -0.09494307405266897 |
| C_N_beta | PASS | -0.14439727153667228 |
| C_Y_r | PASS | -0.35799322284126345 |
| C_L_r | PASS | -0.13515207007914593 |
| C_N_r | PASS | -0.10914614909600459 |
| cg_cruise_fwd_ok | PASS | -2.2548184079243168 |
| cg_cruise_aft_ok | PASS | -0.03989197461127958 |
| cg_vtol_fwd_ok | PASS | -0.10090259012102987 |
| cg_vtol_aft_ok | PASS | -0.31649025018783306 |

### Stability derivatives

| derivative | value |
|---|---|
| C_L_alpha | 6.067317944874764 |
| C_M_alpha | -6.160299353371169 |
| C_L_alpha_dot | 4.305243024403541 |
| C_M_alpha_dot | -16.248992161059586 |
| C_L_q | 13.37616426890122 |
| C_M_q | -104.09487789356604 |
| C_Y_beta | -2.1808877618642937 |
| C_L_beta | -0.09494307405266897 |
| C_N_beta | 0.14439727153667228 |
| C_Y_beta_dot | 0.011428237557844757 |
| C_N_beta_dot | -0.0033114134124902165 |
| C_Y_p | -0.008873205441599571 |
| C_L_p | -0.386320573730939 |
| C_N_p | -0.046901118967110245 |
| C_Y_r | 0.35799322284126345 |
| C_L_r | 0.13515207007914593 |
| C_N_r | -0.10914614909600459 |
| C_X_0 | -0.0 |
| C_Z_0 | -0.592991707858152 |
| C_X_u | -0.02506080955284925 |
| C_Z_u | -1.2037940361705124 |
| C_X_alpha | 0.3365768246512808 |
| C_Z_alpha | -6.079848349651189 |
| C_Z_alpha_dot | -4.305243024403541 |
| C_Z_q | -13.37616426890122 |
| C_L_delta_e | 0.7996507891170686 |
| C_M_delta_e | -3.01806874322699 |
| C_Z_delta_e | -0.7996507891170686 |
| C_X_delta_e | 0.0 |
| C_Y_delta_r | 0.14143668226174286 |
| C_N_delta_r | -0.040982288326528206 |
| C_L_delta_r | 0.010039871031539407 |
| C_L_delta_a | 0.03618434703729835 |
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
| static_long_stab | missing dependency: plotly |
| vtol_oei | missing dependency: plotly |
| dyn_stab | failed: module 'control' has no attribute 'ss' |
| landing_gear_trade | ok |

## Parameter list

Full parameter dump: `data/parameters_full.txt` / `.json` (234 module-level constants + the AircraftParameters dataclass tree).

## Figures

All graphs are consolidated in `figures/`. The Sobol/Tornado sensitivity heatmaps are copied in as standalone artifacts (NOT recomputed by this run).
