"""
Rating scale handling.

Indian credit rating agencies (CRISIL, ICRA, CARE, India Ratings, Acuite,
Brickwork, Infomerics) all use the SEBI-standardised long-term scale:
AAA, AA, A, BBB, BB, B, C, D, with '+' and '-' modifiers from AA to C.

We convert every rating to a "notch" number so that a downgrade is simply
an increase in the notch: AAA = 1 ... D = 20.
"""
import re
from typing import Optional

SCALE = [
    "AAA",
    "AA+", "AA", "AA-",
    "A+", "A", "A-",
    "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-",
    "B+", "B", "B-",
    "C+", "C", "C-",
    "D",
]
NOTCH = {r: i + 1 for i, r in enumerate(SCALE)}
WITHDRAWN = "WR"

# Broad categories used for the transition matrix rows/columns
CATEGORIES = ["AAA", "AA", "A", "BBB", "BB", "B", "C", "D"]
INVESTMENT_GRADE_MAX_NOTCH = NOTCH["BBB-"]  # 10

_AGENCY_PREFIX = re.compile(
    r"^\s*(?:\[\s*(?:ICRA|SYN)\s*\]|CRISIL|ICRA|IND|CARE|BWR|ACUITE|INFOMERICS|IVR|PROVISIONAL|SYN)\s*",
    re.IGNORECASE,
)
_CORE = re.compile(r"^(AAA|AA|A|BBB|BB|B|C|D)\s*([+-])?(?![0-9A-Z])", re.IGNORECASE)


def normalise(raw: Optional[str]) -> Optional[str]:
    """
    Turn an agency rating string into a scale symbol.

    'CRISIL AA+/Stable'           -> 'AA+'
    '[ICRA]A- (Negative)'         -> 'A-'
    'IND BBB+ (CE)'               -> 'BBB+'
    'CARE D'                      -> 'D'
    'Withdrawn'                   -> 'WR'
    'CRISIL A1+' (short-term)     -> None   (short-term scale is out of scope)
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    if "withdraw" in s.lower() or s.upper() == WITHDRAWN:
        return WITHDRAWN
    # Remove agency prefixes (may repeat, e.g. "Provisional CARE AA")
    prev = None
    while prev != s:
        prev = s
        s = _AGENCY_PREFIX.sub("", s)
    m = _CORE.match(s)
    if not m:
        return None
    sym = m.group(1).upper() + (m.group(2) or "")
    if sym in ("AAA+", "AAA-", "D+", "D-"):
        sym = sym[:-1]
    return sym if sym in NOTCH else None


def notch(rating: Optional[str]) -> Optional[int]:
    sym = normalise(rating)
    return NOTCH.get(sym) if sym else None


def category(rating: Optional[str]) -> Optional[str]:
    """Broad category: 'AA-' -> 'AA', 'D' -> 'D', 'WR' -> 'WR'."""
    sym = normalise(rating)
    if sym is None:
        return None
    if sym == WITHDRAWN:
        return WITHDRAWN
    return sym.rstrip("+-")


def is_investment_grade(rating: Optional[str]) -> Optional[bool]:
    n = notch(rating)
    return None if n is None else n <= INVESTMENT_GRADE_MAX_NOTCH


def classify_action(rating_from: Optional[str], rating_to: Optional[str]) -> str:
    """Downgrade / upgrade / reaffirm / default / withdrawn / new, decided by notches, not by words."""
    to_sym = normalise(rating_to)
    from_sym = normalise(rating_from)
    if to_sym == WITHDRAWN:
        return "withdrawn"
    if to_sym is None:
        return "unknown"
    if from_sym is None or from_sym == WITHDRAWN:
        return "assigned"
    if to_sym == "D" and from_sym != "D":
        return "default"
    diff = NOTCH[to_sym] - NOTCH[from_sym]
    if diff > 0:
        return "downgrade"
    if diff < 0:
        return "upgrade"
    return "reaffirm"
