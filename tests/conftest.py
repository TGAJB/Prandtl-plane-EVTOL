"""
Shared pytest configuration for the V&V suite.

Forces matplotlib onto the non-interactive 'Agg' backend before any test
module imports the performance-diagram scripts (vn_diagram, turnperformance,
climb, payloadrange), which import matplotlib.pyplot at module level. This
keeps the suite headless and prevents a figure window / display dependency
during automated regression runs (system test S2).
"""

import matplotlib

matplotlib.use("Agg")
