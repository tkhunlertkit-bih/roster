# 🚦🏥 Nurse Traffic Control

**A constraint-based nurse scheduling optimization system for hospital
inpatient departments.**

This solver generates monthly nurse rosters that satisfy complex staffing
requirements, respect nurse preferences, and maintain safe patient-to-staff
ratios using Google OR-Tools CP-SAT.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

### Hard Constraints (Must Be Satisfied)

The solver enforces the following hard constraints using constraint programming:

**1. One Shift Per Nurse Per Day**

Each nurse is assigned exactly one shift (including off-days) per day.

```math
\sum_{s \in S} x_{n,d,s} = 1 \quad \forall n \in N, d \in D
```

**2. Minimum Daily Staffing**

At least one working nurse (non-off shift) must be scheduled each day.

```math
\sum_{n \in N} \sum_{s \in S_{\text{work}}} x_{n,d,s} \ge 1 \quad \forall d \in D
```

**3. RN/PN Skill Mix Per Shift**

Maintains registered nurse (RN) to practical nurse (PN) ratio between 35%-65% for each shift each day when staff is assigned.

```math
0.35 \le \frac{\text{RN count}}{\text{RN count} + \text{PN count}} \le 0.65
```

**4. Block-Based Staff-to-Patient Ratios**

Enforces safe staffing levels for four time blocks (B1: 07-15, B2: 15-19, B3: 19-23, B4: 23-07) based on daily bed occupancy:

- **Day blocks (B1, B2, B3):**
  - RN requirement: $\lceil \text{beds} / 5 \rceil$
  - PN requirement: $\lceil \text{beds} / 6 \rceil$

- **Night block (B4):**
  - RN requirement: $\lceil \text{beds} / 6 \rceil$
  - PN requirement: $\lceil \text{beds} / 7 \rceil$

```math
\sum_{\substack{n \in N \\ \text{skill}(n)=k}} \sum_{s \in S} C_{s,b} \, x_{n,d,s} \ge \text{Req}_{d,b,k} \quad \forall d,b,k
```

where $C_{s,b} \in \{0,1\}$ indicates whether shift $s$ covers block $b$.

**5. Monthly Hours Limits**

Each nurse must work within their contracted monthly hours range.

```math
L_n \le \sum_{d \in D} \sum_{s \in S} H_s \, x_{n,d,s} \le U_n \quad \forall n
```

**6. Forbidden Shift Transitions**

Prevents dangerous consecutive shift patterns (e.g., evening → morning, night → day shifts):

```math
x_{n,d,s} + x_{n,d+1,s'} \le 1 \quad \forall n,d,\ (s,s') \in \text{ForbiddenNext}
```

Default forbidden transitions:

- E (evening) → {M, M4, ME}
- N (night) → {M, M4, ME, E}
- N4 (12h night) → {M, M4, ME, E}
- ME (16h shift) → {M, M4, ME}

**7. Maximum Consecutive Working Days**

After 4 consecutive working days, the next day must be off.

```math
\sum_{k=0}^{3} \mathbb{1}[\text{nurse } n \text{ works on day } d+k] = 4 \implies x_{n,d+4,s} = 0 \quad \forall s \in S_{\text{work}}
```

**8. Leave Constraints**

- **Birthday leave (BD):** Maximum of `MAX_BD_PER_MONTH` days per nurse (default: 0)
- **Vacation (V):** Only assigned when explicitly requested
- **Public holidays (PUB):** Equalized across all nurses (same count per nurse)

```math
\sum_{d \in D} x_{n,d,\text{PUB}} = \text{pub\_days\_per\_nurse} \quad \forall n
```

### Soft Constraints (Optimized)

The solver minimizes a weighted objective function:

```math
\text{Obj} = w_{\text{hours}} \sum_n \text{excess}_n
+ w_{\text{dayoff}} \sum_n \text{deniedX}_n
+ w_{\text{pref}} \sum_n \text{penalty}_n
+ w_{\text{fair}} \sum_n |\text{penalty}_n - \bar{P}|
+ w_{\text{fte}} \max(0, \text{total\_hours} - \text{target})
```

where:

- $\text{excess}_n = \max(0, \text{hours}_n - 184)$: hours beyond baseline
- $\text{deniedX}_n$: count of denied day-off requests (R or X preferences)
- $\text{penalty}_n$: count of other unmatched shift preferences
- $\bar{P}$: average penalty across nurses with preferences
- Target FTE-UOS hours: $\text{numberOfBeds} \times \text{FTE threshold} \times 8$

**Default weights** (configurable in `configs/ipd_nurse.py`):

- `DENIED_DAYS_OFF_WEIGHT`: 100,000 (highest priority)
- `EXCEEDING_HOURS_WEIGHT`: 10,000
- `PREFERENCE_WEIGHT`: 8,000
- `FAIRNESS_WEIGHT`: 5,000
- `PREF_REWARD`: 1,000 (bonus for matching long-shift preferences)
- `OVER_FTE_WEIGHT`: 1

### Advanced Features

- **Two-stage lexicographic optimization:** Stage 1 minimizes denied day-off requests; Stage 2 optimizes other preferences while maintaining Stage 1 quality
- **Multi-seed random restart:** Runs multiple random seeds and selects the best feasible solution
- **Post-optimization swaps:** Improves solutions via:
  - Day-level swaps: exchange shifts between same-skill nurses on individual days
  - Month-level swaps: exchange entire monthly patterns between same-skill nurses
- **Comprehensive validation:** Verifies all hard constraints after solving
- **Detailed reporting:** Generates multi-sheet Excel output with rosters, statistics, block coverage, and preference analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Setup

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/nurse-rostering.git
cd nurse-rostering
```

2. **Create a conda environment:**

```bash
conda create -n nurse-roster python=3.10
conda activate nurse-roster
```

3. **Install dependencies:**

```bash
python -m pip install -r requirements.txt
```

## Input Data Format

All input files must be in CSV format and placed in a directory (e.g., `input_data/ipd_nurse/`).

### 1. `nurses.csv`

Contains nurse information and monthly hour constraints.

**Required columns:**

- `nurse_id` (str): Unique identifier for each nurse
- `name` (str): Nurse's full name
- `phone_number` (str): Contact number
- `skill` (str): Qualification level - must be one of: `RN` (Registered Nurse), `PN` (Practical Nurse), `SUP` (Supervisor)
- `monthly_min_hours` (int): Minimum contracted hours for the month
- `monthly_max_hours` (int): Maximum allowed hours for the month
- `level` (str, optional): Seniority level (e.g., "N1", "N2", "Senior")

**Example:**

```csv
nurse_id,name,phone_number,skill,monthly_min_hours,monthly_max_hours,level
N001,Alice Johnson,555-0101,RN,160,200,N2
N002,Bob Smith,555-0102,PN,160,192,N1
N003,Carol White,555-0103,RN,184,220,Senior
N004,David Lee,555-0104,PN,160,192,N1
```

### 2. `preferences.csv`

Contains shift preferences for each nurse for each day of the month.

**Required columns:**

- `nurse_id` (str): Must match IDs in `nurses.csv`
- `1`, `2`, `3`, ..., `31` (str): Preference for each day (up to the number of days in the month)

**Preference values:**

- Shift codes: `M` (morning), `E` (evening), `N` (night), `M4` (12h morning), `N4` (12h night), `ME` (16h)
- `X` or `R`: Request day off
- `V`: Request vacation (leave)
- Empty or blank: No preference for that day

**Example:**

```csv
nurse_id,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
N001,M,M,X,M,M,,,E,E,E,R,E,N,N,V
N002,,,M4,M4,,X,M,M,M,M,,,N4,N4,
N003,N,N,N,X,E,E,E,M,M,X,M,M,,,E
N004,M4,M4,R,M,M,M,X,E,E,E,E,N,N,N,X
```

### 3. `beds_per_day.csv`

Contains daily bed occupancy to calculate staff-to-patient ratios.

**Required columns:**

- `day` (int): Day number (1 to number of days in month)
- `beds` (int): Number of occupied beds for that day

**Example:**

```csv
day,beds
1,45
2,47
3,46
4,44
5,48
6,50
7,49
```

## Configuration

The solver uses modular configuration files in the `configs/` directory. Each configuration defines department-specific rules.

### Creating a Configuration

Example: `configs/ipd_nurse.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Set

# Time blocks for coverage
TIME_BLOCKS = ["B1", "B2", "B3", "B4"]
BLOCK_LABELS = {
    "B1": "07:00-15:00",
    "B2": "15:00-19:00",
    "B3": "19:00-23:00",
    "B4": "23:00-07:00",
}

# Shift definitions
@dataclass(frozen=True)
class ShiftDef:
    name: str
    hours: int
    blocks: List[str]

SHIFT_DEFS = [
    ShiftDef("X", 0, []),           # Day off
    ShiftDef("M", 8, ["B1"]),       # Morning
    ShiftDef("E", 8, ["B2", "B3"]), # Evening
    ShiftDef("N", 8, ["B4"]),       # Night
    ShiftDef("M4", 12, ["B1", "B2"]),
    ShiftDef("N4", 12, ["B3", "B4"]),
    ShiftDef("ME", 16, ["B1", "B2", "B3"]),
    ShiftDef("V", 8, []),           # Vacation
    ShiftDef("BD", 8, []),          # Birthday leave
    ShiftDef("PUB", 8, []),         # Public holiday
]

# Shift categories
OFF_SHIFTS = {"X", "V", "BD", "PUB"}
LONG_SHIFTS = ["M4", "N4", "ME"]

# Forbidden transitions
FORBIDDEN_NEXT = {
    "E": {"M", "M4", "ME"},
    "N": {"M", "M4", "ME", "E"},
    "N4": {"M", "M4", "ME", "E"},
    "ME": {"M", "M4", "ME"},
}

# Work rules
MAX_CONSEC_WORK_DAYS = 4
MAX_BD_PER_MONTH = 0

# Objective weights
EXCEEDING_HOURS_WEIGHT = 10_000
DENIED_DAYS_OFF_WEIGHT = 100_000
PREFERENCE_WEIGHT = 8_000
FAIRNESS_WEIGHT = 5_000
OVER_FTE_WEIGHT = 1
LONG_SHIFT_PENALTY = 0
PREF_REWARD = 1_000
```

## Usage

### Basic Command

```bash
python solver.py \
  --config ipd_nurse \
  --input_dir input_data/ipd_nurse \
  --pub_days_per_nurse 1 \
  --fte_uos_threshold 1.9 \
  --days 28
```

### Arguments

- `--config`: Name of configuration file in `configs/` (without `.py` extension)
- `--input_dir`: Directory containing input CSV files
- `--pub_days_per_nurse`: Number of public holiday allowed for each nurse this month
- `--fte_uos_threshold`: FTE-UOS ratio threshold for soft hours cap (monthly hours target = beds_sum × threshold × 8)
- `--days`: Number of days in the month (28, 29, 30, or 31)
- `--maxtime`: Maximum solver time in seconds per stage (default: 60)

### Example Scenarios

**February (28 days, 2 public holidays):**

```bash
python solver.py \
  --config ipd_nurse \
  --input_dir input_data/ipd_nurse \
  --pub_days_per_nurse 2 \
  --fte_uos_threshold 1.85 \
  --days 28
```

**December (31 days, 1 public holiday, higher workload):**

```bash
python solver.py \
  --config ipd_nurse \
  --input_dir input_data/ipd_nurse \
  --pub_days_per_nurse 1 \
  --fte_uos_threshold 2.1 \
  --days 31 \
  --maxtime 120
```

## Output

The solver generates `roster_output.xlsx` in the input directory with multiple sheets:

### Sheet 1: `Roster`

Main roster table with nurse assignments.

**Columns:**

- `name_phone`: Nurse name and phone number
- `level`: Seniority level
- `role`: Skill type (RN/PN)
- `nurse_id`: Unique identifier
- `threshold_hours`: Expected baseline hours (184)
- `total_hours`: Actual scheduled hours
- `exceed_hours`: Hours above baseline (max(0, total - 184))
- `flag_over_240`: Boolean flag if exceed_hours > 240
- `1`, `2`, `3`, ..., `{days}`: Assigned shift for each day

**Example row:**

```
Alice Johnson (555-0101) | N2 | RN | N001 | 184 | 192 | 8 | 0 | M | M | X | E | ...
```

### Sheet 2: `BlockCoverage`

Verifies staff-to-patient ratios for each time block.

**Columns per day:**

- `day`: Day number
- `block`: Time block (B1/B2/B3/B4)
- `total`: Total staff covering this block
- `RN`: Count of RNs
- `PN`: Count of PNs

### Sheet 3: `ShiftStatistics`

Count of each shift type per day.

**Example:**

```
Shift | Count
------|------
M     | 45
E     | 38
N     | 22
M4    | 15
X     | 30
```

### Sheet 4: `request summary`

Per-nurse preference satisfaction statistics.

**Columns:**

- `nurse_id`
- `total_requests`: Number of days with preferences
- `accepted`: Requests granted
- `denied`: Requests not granted

### Sheet 5: `denied requests`

Detailed log of each denied preference.

**Columns:**

- `nurse_id`
- `day`: Day number
- `requested`: Preferred shift
- `assigned`: Actual shift assigned

## Algorithm Overview

1. **Multi-seed optimization:** Runs 10 independent solver attempts with different random seeds
2. **Two-stage lexicographic solve:**
   - **Stage 1:** Minimize denied day-off requests (X/R) + excess hours
   - **Stage 2:** Fix day-off denials at Stage 1 level, optimize shift preferences + fairness + FTE-UOS
3. **Post-processing:**
   - Month-level swaps between same-skill nurses
   - Day-level swaps to improve preference matches
4. **Validation:** Ensures all hard constraints are satisfied
5. **Best selection:** Returns the solution with lowest overall score

## Project Structure

```
nurse-rostering/
├── solver.py              # Main solver entry point
├── validate_roster.py     # Hard constraint validation
├── swap_utils.py          # Post-optimization swap heuristics
├── requirements.txt       # Python dependencies
├── configs/
│   └── ipd_nurse.py      # IPD nurse ward configuration
└── input_data/
    └── ipd_nurse/
        ├── nurses.csv
        ├── preferences.csv
        ├── beds_per_day.csv
        └── roster_output.xlsx  (generated)
```

## Requirements

- `ortools`: Google's constraint programming solver
- `pandas`: Data manipulation
- `openpyxl`: Excel file writing
- `xlsxwriter`: Excel formatting

## Extending to Other Departments

To adapt this solver for a different department:

1. Create a new config file: `configs/your_department.py`
2. Define department-specific:
   - Shift types and hours
   - Time blocks and coverage rules
   - Forbidden transitions
   - Objective weights
3. Update staff-to-patient ratio functions if needed (in `solver.py`)
4. Run with `--config your_department`

## Troubleshooting

**No feasible solution:**

- Check if monthly hours ranges are realistic for the number of nurses and beds
- Verify public holiday count matches available staff
- Review forbidden transitions—some combinations may be too restrictive
- Increase `--maxtime` for longer solving

**Poor preference satisfaction:**

- Increase `PREFERENCE_WEIGHT` in config
- Reduce `FAIRNESS_WEIGHT` if over-constraining
- Check if nurse requests conflict with coverage needs

**Validation errors:**

- Review `beds_per_day.csv` for unrealistic bed counts
- Ensure all nurse IDs match between CSVs
- Check skill distribution (need both RN and PN for ratios)

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional shift patterns and constraints
- Performance optimizations
- Web UI for easier input/visualization
- Integration with hospital management systems

## Acknowledgments

- Built with [Google OR-Tools](https://developers.google.com/optimization)
- Constraint programming approach inspired by nurse rostering literature
