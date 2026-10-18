"""
SpreadWatch - central configuration.

Every tunable number in the system lives here, so that the assumptions
defended in assumptions.md map one-to-one to a line of code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# Data locations
# ----------------------------------------------------------------------------
RAW_DIR = ROOT / "data" / "raw"            # files you download by hand go here
SAMPLE_DIR = ROOT / "data" / "sample"      # SYNTHETIC demo data (never used for findings)
REAL_DIR = ROOT / "data" / "real"          # canonical CSVs produced by ingest.py from RAW_DIR
OUTPUT_DIR = ROOT / "outputs"
SOURCES_CSV = ROOT / "sources.csv"

# Canonical file names (same names in SAMPLE_DIR and REAL_DIR)
TRADES_FILE = "trades.csv"
CURVE_FILE = "gsec_curve.csv"
RATINGS_FILE = "rating_actions.csv"

# ----------------------------------------------------------------------------
# Bond maths (assumptions.md A1-A4)
# ----------------------------------------------------------------------------
FACE_VALUE = 100.0                 # prices are quoted per 100 of face value
DAY_COUNT_BASIS = 365.0            # Actual/365 for accrued interest and time-to-cash-flow
DEFAULT_COUPON_FREQ = 1            # most Indian corporate bonds pay annually
GSEC_COMPOUNDING = 2               # FBIL G-sec par yields are semi-annual
YIELD_TOLERANCE_BPS = 5.0          # validation: |computed - printed| must be within this

# Spread basis: "annual_effective" converts the semi-annual G-sec yield to an
# annual effective rate before subtracting it from the (annualised, XIRR-style)
# corporate yield. "quoted" subtracts the two quoted yields directly.
SPREAD_BASIS = "annual_effective"

# Curve look-up: use the latest curve on or before the trade date, but never
# one older than this many calendar days.
MAX_CURVE_STALENESS_DAYS = 7

# ----------------------------------------------------------------------------
# Event study (assumptions.md A6-A10)
# ----------------------------------------------------------------------------
BASELINE_WINDOW = (-180, -91)      # calendar days relative to the rating action
PRE_WINDOW = (-30, -1)             # "just before the agency acts"
DETECTION_WINDOW = (-90, -1)       # where we look for the first early-warning breach
POST_WINDOW = (0, 30)              # reaction after the action
Z_THRESHOLD = 2.0                  # breach = abnormal spread > baseline mean + 2 sd
MIN_BASELINE_OBS = 4               # fewer observations -> event is excluded
MIN_PRE_OBS = 2
EVENT_CLUSTER_DAYS = 180           # later downgrades of the same issuer within this many
                                   # days are treated as part of the same episode
N_PLACEBO_PER_ISSUER = 5           # random non-event dates per control issuer
RANDOM_SEED = 42
SIGNIFICANCE = 0.05

# ----------------------------------------------------------------------------
# Transition matrix / PD (assumptions.md A11-A13)
# ----------------------------------------------------------------------------
COHORT_MONTH_DAY = (4, 1)          # cohorts formed on 1 April (Indian financial year)
HORIZON_YEARS = 1
LGD = 0.45                         # Basel / RBI foundation-IRB LGD for senior unsecured claims
