# NASA/Langley LS(1)-0417 Airfoil Validation Summary

AirfoilTools is treated as the reference dataset. Flow5 values are linearly interpolated onto the AirfoilTools angle-of-attack grid inside the overlapping alpha range.

## Dataset metadata

| Item | Value |
| --- | --- |
| Reference | AirfoilTools XFOIL 6.96 |
| Candidate | Flow5 v7.56 |
| Reference file | ls10417_airfoiltools_Re1e6_N9.txt |
| Candidate file | ls10417_flow5_Re1e6_N9.txt |
| Airfoil | NASA/LANGLEY LS(1)-0417 (GA(W)-1) AIRFOIL |
| Reynolds number | 1000000.0000 |
| Mach | 0.0000 |
| Ncrit | 9.0000 |
| Comparison alpha range | -18.30 to 19.25 deg |
| Linear fit alpha range | -2.00 to 6.00 deg |

## Aerodynamic characteristics

| Metric | AirfoilTools | Flow5 | Flow5 - AirfoilTools |
| --- | --- | --- | --- |
| CL_at_0deg | 0.53560 | 0.53230 | -0.00330 |
| CD_at_0deg | 0.00695 | 0.00691 | -0.00004 |
| Cm_at_0deg | -0.11840 | -0.11780 | 0.00060 |
| CL_max | 1.78550 | 1.79150 | 0.00600 |
| alpha_at_CL_max_deg | 18.25000 | 18.00000 | -0.25000 |
| CD_min | 0.00658 | 0.00663 | 0.00005 |
| alpha_at_CD_min_deg | -3.25000 | -3.20000 | 0.05000 |
| LD_max | 112.61808 | 109.36314 | -3.25494 |
| alpha_at_LD_max_deg | 2.50000 | 2.30000 | -0.20000 |
| lift_curve_slope_per_deg | 0.10834 | 0.10779 | -0.00055 |
| alpha_zero_lift_deg | -4.94378 | -4.93504 | 0.00874 |

## Pointwise error summary

| Coeff. | MAE | RMSE | Max abs. | Max abs. alpha [deg] | MAE / ref. range [%] |
| --- | --- | --- | --- | --- | --- |
| CL | 0.00691 | 0.01090 | 0.07045 | -18.25000 | 0.24890 |
| CD | 0.00075 | 0.00196 | 0.01478 | -18.25000 | 0.89354 |
| CDp | 0.00076 | 0.00204 | 0.01520 | -18.25000 | 0.89709 |
| Cm | 0.00097 | 0.00166 | 0.01045 | -18.25000 | 1.49284 |
| Top_Xtr | 0.00554 | 0.01004 | 0.04880 | 3.00000 | 0.57132 |
| Bot_Xtr | 0.00383 | 0.00801 | 0.07300 | 18.25000 | 0.38886 |

## Key alpha differences

| Alpha [deg] | Delta CL | Delta CD [counts] | Delta Cm |
| --- | --- | --- | --- |
| -4.0000 | -0.0032 | 0.7000 | 0.0009 |
| 0.0000 | -0.0033 | -0.4000 | 0.0006 |
| 2.0000 | -0.0040 | 0.1000 | 0.0007 |
| 4.0000 | -0.0075 | 3.1000 | 0.0012 |
| 6.0000 | -0.0068 | 2.2000 | 0.0011 |
| 8.0000 | -0.0067 | 2.5000 | 0.0010 |
| 10.0000 | -0.0040 | 1.0000 | 0.0005 |
| 12.0000 | -0.0065 | 2.7000 | 0.0008 |
| 14.0000 | -0.0003 | -1.6000 | 0.0001 |
| 16.0000 | 0.0024 | -4.6000 | 0.0000 |
| 18.0000 | 0.0063 | -10.3000 | 0.0000 |

Note: this is a validation against another XFOIL-based reference calculation, not against wind-tunnel data. Agreement should be described as code-to-reference consistency.

