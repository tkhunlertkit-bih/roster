# config.py

from dataclasses import dataclass
from typing import Dict, List, Set

# ---- time blocks -----------------------------------------------------------

# Ordered blocks in the day; used to enforce coverage.
TIME_BLOCKS = ["B1", "B2", "B3", "B4"]
BLOCK_LABELS = {
    "B1": "07:00-15:00",
    "B2": "15:00-19:00",
    "B3": "19:00-23:00",
    "B4": "23:00-07:00",
}

# ---- shifts and hours ------------------------------------------------------


@dataclass(frozen=True)
class ShiftDef:
    name: str
    hours: int
    blocks: List[str]  # which blocks this shift covers


# department-specific shift settings (IPD example)
SHIFT_DEFS: List[ShiftDef] = [
    ShiftDef("X", 0, []),
    ShiftDef("M", 8, ["B1"]),
    ShiftDef("E", 8, ["B2", "B3"]),
    ShiftDef("N", 8, ["B4"]),
    ShiftDef("M4", 12, ["B1", "B2"]),
    ShiftDef("N4", 12, ["B3", "B4"]),
    ShiftDef("ME", 16, ["B1", "B2", "B3"]),
    ShiftDef("V", 8, []),  # leave: no block coverage
    ShiftDef("BD", 8, []),
    ShiftDef("PUB", 8, []),
]

# derived lookups
SHIFT_NAMES = [s.name for s in SHIFT_DEFS]
SHIFT_HOURS = {s.name: s.hours for s in SHIFT_DEFS}
SHIFT_BLOCKS = {s.name: list(s.blocks) for s in SHIFT_DEFS}

# ---- skills / nurse types --------------------------------------------------

# basic: RN vs PN (can extend to WS, etc.)
SKILLS = ["RN", "PN"]
SKILL_PROPORTION = [60, 40]

# The nurse CSV should have a 'skill' column with one of SKILLS.

# ---- forbidden adjacent transitions ----------------------------------------

# one-way forbidden transitions for this department
FORBIDDEN_NEXT: Dict[str, Set[str]] = {
    "E": {"M", "M4", "ME"},
    "N": {"M", "M4", "ME", "E"},
    "N4": {"M", "M4", "ME", "E"},
    "ME": {"M", "M4", "ME"},
    # M and M4: no restrictions
}

# ---- days-off / rest configuration -----------------------------------------

OFF_SHIFTS = {"X", "V", "BD", "PUB"}

MAX_OFF_IN_4_DAYS = 3  # max 3 days off in any 4-day window
MAX_CONSEC_WORK_DAYS = 4  # after 4 workdays, next day must be off

# ---- leave limits ----------------------------------------------------------

MAX_BD_PER_MONTH = 1
EQUALIZE_PUB_COUNTS = True

# ---- objective weighting ----------------------------------------------------

PREFERENCE_WEIGHT = 1
FAIRNESS_WEIGHT = 1
