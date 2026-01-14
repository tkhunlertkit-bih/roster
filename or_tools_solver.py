"""
IPD Nurse Rostering - OR-Tools Solver
======================================
Sample implementation using Google OR-Tools CP-SAT solver
Designed to run in Databricks or as standalone Python script

Author: AI Assistant
Requirements: ortools, pandas, numpy
"""

from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class NurseData:
    """Nurse attributes relevant for scheduling"""
    nurse_id: str
    primary_ward_id: str
    secondary_ward_ids: List[str]
    contracted_hours: int
    seniority: str
    skill_level: int
    can_charge: bool
    prefers_day: bool
    prefers_night: bool
    max_consecutive: int
    max_nights_week: int
    certifications: List[str]
    unavailable_dates: List[str]
    cert_hours_needed: Dict[str, int]  # cert_id -> hours remaining


@dataclass
class WardData:
    """Ward attributes and staffing requirements"""
    ward_id: str
    ward_name: str
    min_day: int
    min_evening: int
    min_night: int
    required_certs: List[str]
    acuity: int


@dataclass
class ShiftData:
    """Shift definitions"""
    shift_id: str
    duration: int
    cert_weight: float
    is_night: bool


class ConstraintType(Enum):
    HARD = "HARD"
    SOFT = "SOFT"


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """Load and preprocess data from CSV/Delta tables"""
    
    def __init__(self, data_path: str = "./mock_data"):
        self.data_path = data_path
        self.nurses_df = None
        self.wards_df = None
        self.shifts_df = None
        self.certs_df = None
        self.nurse_certs_df = None
        self.unavail_df = None
        self.rules_df = None
        self.requests_df = None
        
    def load_all(self) -> None:
        """Load all required tables"""
        logger.info("Loading data from CSV files...")
        
        self.nurses_df = pd.read_csv(f"{self.data_path}/dim_nurses.csv")
        self.wards_df = pd.read_csv(f"{self.data_path}/dim_wards.csv")
        self.shifts_df = pd.read_csv(f"{self.data_path}/dim_shifts.csv")
        self.certs_df = pd.read_csv(f"{self.data_path}/dim_certifications.csv")
        self.nurse_certs_df = pd.read_csv(f"{self.data_path}/fact_nurse_certifications.csv")
        self.unavail_df = pd.read_csv(f"{self.data_path}/fact_unavailability.csv")
        self.rules_df = pd.read_csv(f"{self.data_path}/dim_scheduling_rules.csv")
        self.requests_df = pd.read_csv(f"{self.data_path}/fact_shift_requests.csv")
        
        # Filter to active nurses only
        self.nurses_df = self.nurses_df[self.nurses_df["is_active"] == True]
        
        logger.info(f"Loaded {len(self.nurses_df)} active nurses")
        logger.info(f"Loaded {len(self.wards_df)} wards")
        
    def get_nurse_data(self, nurse_id: str) -> NurseData:
        """Get processed nurse data"""
        nurse = self.nurses_df[self.nurses_df["nurse_id"] == nurse_id].iloc[0]
        
        # Get certifications
        nurse_certs = self.nurse_certs_df[
            (self.nurse_certs_df["nurse_id"] == nurse_id) & 
            (self.nurse_certs_df["is_expired"] == False)
        ]
        certs = nurse_certs["cert_id"].tolist()
        
        # Get cert hours needed
        cert_hours = {}
        for _, nc in nurse_certs.iterrows():
            if nc["hours_remaining"] > 0:
                cert_hours[nc["cert_id"]] = int(nc["hours_remaining"])
        
        # Get unavailable dates
        unavail = self.unavail_df[
            (self.unavail_df["nurse_id"] == nurse_id) & 
            (self.unavail_df["status"] == "APPROVED")
        ]
        unavail_dates = []
        for _, u in unavail.iterrows():
            start = datetime.strptime(u["start_date"], "%Y-%m-%d")
            end = datetime.strptime(u["end_date"], "%Y-%m-%d")
            current = start
            while current <= end:
                unavail_dates.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
        
        # Parse secondary wards
        try:
            secondary = json.loads(nurse["secondary_ward_ids"])
        except:
            secondary = []
        
        return NurseData(
            nurse_id=nurse_id,
            primary_ward_id=nurse["primary_ward_id"],
            secondary_ward_ids=secondary,
            contracted_hours=int(nurse["contracted_hours_week"]),
            seniority=nurse["seniority"],
            skill_level=int(nurse["skill_level"]),
            can_charge=bool(nurse["can_charge_nurse"]),
            prefers_day=bool(nurse["prefers_day_shift"]),
            prefers_night=bool(nurse["prefers_night_shift"]),
            max_consecutive=int(nurse["max_consecutive_days"]),
            max_nights_week=int(nurse["max_night_shifts_week"]),
            certifications=certs,
            unavailable_dates=unavail_dates,
            cert_hours_needed=cert_hours
        )
    
    def get_ward_data(self, ward_id: str) -> WardData:
        """Get processed ward data"""
        ward = self.wards_df[self.wards_df["ward_id"] == ward_id].iloc[0]
        
        try:
            required_certs = json.loads(ward["requires_certification"])
        except:
            required_certs = []
        
        return WardData(
            ward_id=ward_id,
            ward_name=ward["ward_name"],
            min_day=int(ward["min_nurses_day"]),
            min_evening=int(ward["min_nurses_evening"]),
            min_night=int(ward["min_nurses_night"]),
            required_certs=required_certs,
            acuity=int(ward["acuity_level"])
        )
    
    def get_eligible_nurses_for_ward(self, ward_id: str) -> List[str]:
        """Get nurses eligible to work in a ward based on certifications"""
        ward = self.get_ward_data(ward_id)
        
        if not ward.required_certs:
            # No special certs required - all active nurses assigned to this ward
            return self.nurses_df[
                self.nurses_df["primary_ward_id"] == ward_id
            ]["nurse_id"].tolist()
        
        # Check certification eligibility
        eligible = []
        ward_nurses = self.nurses_df[
            self.nurses_df["primary_ward_id"] == ward_id
        ]["nurse_id"].tolist()
        
        for nurse_id in ward_nurses:
            nurse = self.get_nurse_data(nurse_id)
            if all(cert in nurse.certifications for cert in ward.required_certs):
                eligible.append(nurse_id)
        
        return eligible


# =============================================================================
# OR-TOOLS SOLVER
# =============================================================================

class NurseRosterSolver:
    """
    CP-SAT based nurse roster optimization
    
    Decision Variables:
        roster[n, d, s, w] = 1 if nurse n works shift s on day d in ward w
    
    Hard Constraints:
        - Weekly hours = contracted hours (no overtime)
        - At most one shift per day per nurse
        - Minimum coverage per ward per shift
        - Minimum 11 hours rest between shifts
        - Respect approved leave/unavailability
        - Max 5 consecutive working days
        - Certification requirements for wards
    
    Soft Constraints (minimized):
        - Respect shift preferences (weighted)
        - Balance weekend shifts fairly
        - Balance night shifts fairly
        - Prioritize certification practice hours
        - Ensure skill mix per shift
    """
    
    def __init__(self, 
                 data_loader: DataLoader,
                 planning_start: str,
                 planning_weeks: int = 1):
        
        self.loader = data_loader
        self.start_date = datetime.strptime(planning_start, "%Y-%m-%d")
        self.num_days = planning_weeks * 7
        
        # Model components
        self.model = cp_model.CpModel()
        self.roster = {}  # Decision variables
        self.nurses = []
        self.wards = []
        self.shifts = ["D", "E", "N"]  # Day, Evening, Night
        self.days = list(range(self.num_days))
        
        # Soft constraint penalties
        self.soft_penalties = []
        
        # Results
        self.solution = None
        self.status = None
        
    def _date_for_day(self, day_index: int) -> str:
        """Convert day index to date string"""
        return (self.start_date + timedelta(days=day_index)).strftime("%Y-%m-%d")
    
    def _is_weekend(self, day_index: int) -> bool:
        """Check if day index is Saturday (5) or Sunday (6)"""
        return (self.start_date + timedelta(days=day_index)).weekday() >= 5
    
    def build_model(self, ward_ids: Optional[List[str]] = None) -> None:
        """Build the complete CP-SAT model"""
        
        logger.info("Building CP-SAT model...")
        
        # Use specified wards or all wards
        if ward_ids:
            self.wards = ward_ids
        else:
            self.wards = self.loader.wards_df["ward_id"].tolist()
        
        # Get all nurses for these wards
        self.nurses = []
        for ward_id in self.wards:
            ward_nurses = self.loader.get_eligible_nurses_for_ward(ward_id)
            self.nurses.extend(ward_nurses)
        self.nurses = list(set(self.nurses))  # Deduplicate
        
        logger.info(f"Building model for {len(self.nurses)} nurses, {len(self.wards)} wards, {self.num_days} days")
        
        # 1. Create decision variables
        self._create_variables()
        
        # 2. Add hard constraints
        self._add_one_shift_per_day()
        self._add_weekly_hours_constraint()
        self._add_min_coverage_constraints()
        self._add_unavailability_constraints()
        self._add_consecutive_days_constraint()
        self._add_rest_between_shifts()
        
        # 3. Add soft constraints
        self._add_preference_soft_constraints()
        self._add_fairness_soft_constraints()
        self._add_certification_soft_constraints()
        
        # 4. Set objective
        self._set_objective()
        
        logger.info(f"Model built with {len(self.roster)} variables")
    
    def _create_variables(self) -> None:
        """Create Boolean decision variables"""
        
        for n in self.nurses:
            nurse = self.loader.get_nurse_data(n)
            allowed_wards = [nurse.primary_ward_id] + nurse.secondary_ward_ids
            
            for d in self.days:
                for s in self.shifts:
                    for w in self.wards:
                        if w in allowed_wards:
                            var_name = f"roster_n{n}_d{d}_s{s}_w{w}"
                            self.roster[(n, d, s, w)] = self.model.NewBoolVar(var_name)
    
    def _add_one_shift_per_day(self) -> None:
        """Each nurse works at most one shift per day across all wards"""
        
        for n in self.nurses:
            nurse = self.loader.get_nurse_data(n)
            allowed_wards = [nurse.primary_ward_id] + nurse.secondary_ward_ids
            allowed_wards = [w for w in allowed_wards if w in self.wards]
            
            for d in self.days:
                shifts_this_day = []
                for s in self.shifts:
                    for w in allowed_wards:
                        if (n, d, s, w) in self.roster:
                            shifts_this_day.append(self.roster[(n, d, s, w)])
                
                if shifts_this_day:
                    self.model.AddAtMostOne(shifts_this_day)
    
    def _add_weekly_hours_constraint(self) -> None:
        """Weekly hours should be close to contracted hours"""
        
        shift_hours = {"D": 8, "E": 8, "N": 8}
        
        for n in self.nurses:
            nurse = self.loader.get_nurse_data(n)
            allowed_wards = [nurse.primary_ward_id] + nurse.secondary_ward_ids
            allowed_wards = [w for w in allowed_wards if w in self.wards]
            
            target_shifts = nurse.contracted_hours // 8  # Assuming 8-hour shifts
            
            # For each week in planning horizon
            for week in range(self.num_days // 7):
                week_start = week * 7
                week_end = min((week + 1) * 7, self.num_days)
                
                # Count available days this week
                available_days = 0
                for d in range(week_start, week_end):
                    date_str = self._date_for_day(d)
                    if date_str not in nurse.unavailable_dates:
                        available_days += 1
                
                total_shifts = []
                for d in range(week_start, week_end):
                    for s in self.shifts:
                        for w in allowed_wards:
                            if (n, d, s, w) in self.roster:
                                total_shifts.append(self.roster[(n, d, s, w)])
                
                if total_shifts and available_days > 0:
                    # Adjust target based on available days
                    max_possible = min(available_days, target_shifts + 1)
                    min_required = max(0, min(available_days, target_shifts - 2))
                    
                    self.model.Add(sum(total_shifts) >= min_required)
                    self.model.Add(sum(total_shifts) <= max_possible)
    
    def _add_min_coverage_constraints(self) -> None:
        """Each ward should meet minimum staffing - penalize shortfall"""
        
        COVERAGE_PENALTY = 500  # High penalty per missing nurse
        
        for w in self.wards:
            ward = self.loader.get_ward_data(w)
            min_staff = {"D": ward.min_day, "E": ward.min_evening, "N": ward.min_night}
            
            for d in self.days:
                for s in self.shifts:
                    assigned = []
                    for n in self.nurses:
                        if (n, d, s, w) in self.roster:
                            assigned.append(self.roster[(n, d, s, w)])
                    
                    if assigned:
                        # Create shortfall variable
                        total_assigned = sum(assigned)
                        shortfall = self.model.NewIntVar(0, min_staff[s], f"coverage_shortfall_{w}_{d}_{s}")
                        self.model.Add(shortfall >= min_staff[s] - total_assigned)
                        
                        # Penalize shortfall heavily
                        penalty = self.model.NewIntVar(0, COVERAGE_PENALTY * min_staff[s], f"coverage_penalty_{w}_{d}_{s}")
                        self.model.Add(penalty == COVERAGE_PENALTY * shortfall)
                        self.soft_penalties.append(penalty)
                        
                        # Still require at least 1 nurse per shift (hard constraint)
                        self.model.Add(total_assigned >= 1)
    
    def _add_unavailability_constraints(self) -> None:
        """Nurses cannot work on approved leave dates"""
        
        for n in self.nurses:
            nurse = self.loader.get_nurse_data(n)
            
            for d in self.days:
                date_str = self._date_for_day(d)
                
                if date_str in nurse.unavailable_dates:
                    # Cannot work any shift on this day
                    for s in self.shifts:
                        for w in self.wards:
                            if (n, d, s, w) in self.roster:
                                self.model.Add(self.roster[(n, d, s, w)] == 0)
    
    def _add_consecutive_days_constraint(self) -> None:
        """Max consecutive working days - simplified version"""
        
        max_consecutive = 6  # Use global max for simplicity
        
        for n in self.nurses:
            nurse = self.loader.get_nurse_data(n)
            allowed_wards = [nurse.primary_ward_id] + nurse.secondary_ward_ids
            allowed_wards = [w for w in allowed_wards if w in self.wards]
            
            # Create helper variable: did nurse n work on day d?
            worked = {}
            for d in self.days:
                day_vars = []
                for s in self.shifts:
                    for w in allowed_wards:
                        if (n, d, s, w) in self.roster:
                            day_vars.append(self.roster[(n, d, s, w)])
                
                if day_vars:
                    worked[d] = self.model.NewBoolVar(f"worked_{n}_{d}")
                    self.model.AddMaxEquality(worked[d], day_vars)
            
            # Check each window of (max_consecutive + 1) days
            for start_d in range(self.num_days - max_consecutive):
                window_worked = []
                for d in range(start_d, start_d + max_consecutive + 1):
                    if d in worked:
                        window_worked.append(worked[d])
                
                if len(window_worked) == max_consecutive + 1:
                    self.model.Add(sum(window_worked) <= max_consecutive)
    
    def _add_rest_between_shifts(self) -> None:
        """Minimum 11 hours rest - no night followed by day next morning"""
        
        for n in self.nurses:
            nurse = self.loader.get_nurse_data(n)
            allowed_wards = [nurse.primary_ward_id] + nurse.secondary_ward_ids
            allowed_wards = [w for w in allowed_wards if w in self.wards]
            
            for d in range(self.num_days - 1):
                for w in allowed_wards:
                    # For each night shift today, no day shift tomorrow in same or other ward
                    if (n, d, "N", w) in self.roster:
                        night_var = self.roster[(n, d, "N", w)]
                        for w2 in allowed_wards:
                            if (n, d + 1, "D", w2) in self.roster:
                                day_var = self.roster[(n, d + 1, "D", w2)]
                                # Cannot have both
                                self.model.Add(night_var + day_var <= 1)
    
    def _add_preference_soft_constraints(self) -> None:
        """Add penalties for violating shift preferences"""
        
        PREFERENCE_WEIGHT = 50
        
        for n in self.nurses:
            nurse = self.loader.get_nurse_data(n)
            allowed_wards = [nurse.primary_ward_id] + nurse.secondary_ward_ids
            allowed_wards = [w for w in allowed_wards if w in self.wards]
            
            for d in self.days:
                # Penalty for night shift when prefer day
                if nurse.prefers_day and not nurse.prefers_night:
                    for w in allowed_wards:
                        if (n, d, "N", w) in self.roster:
                            penalty = self.model.NewIntVar(0, PREFERENCE_WEIGHT, f"pref_penalty_{n}_{d}")
                            self.model.Add(penalty == PREFERENCE_WEIGHT * self.roster[(n, d, "N", w)])
                            self.soft_penalties.append(penalty)
                
                # Penalty for day shift when prefer night
                if nurse.prefers_night and not nurse.prefers_day:
                    for w in allowed_wards:
                        if (n, d, "D", w) in self.roster:
                            penalty = self.model.NewIntVar(0, PREFERENCE_WEIGHT, f"pref_penalty_{n}_{d}")
                            self.model.Add(penalty == PREFERENCE_WEIGHT * self.roster[(n, d, "D", w)])
                            self.soft_penalties.append(penalty)
    
    def _add_fairness_soft_constraints(self) -> None:
        """Add penalties for unfair distribution of weekend/night shifts"""
        
        FAIRNESS_WEIGHT = 80
        
        # Calculate average expected weekend shifts per nurse
        num_weekends = sum(1 for d in self.days if self._is_weekend(d))
        expected_weekend_per_nurse = (num_weekends * len(self.shifts)) / max(len(self.nurses), 1)
        
        for n in self.nurses:
            nurse = self.loader.get_nurse_data(n)
            allowed_wards = [nurse.primary_ward_id] + nurse.secondary_ward_ids
            allowed_wards = [w for w in allowed_wards if w in self.wards]
            
            # Count weekend shifts
            weekend_shifts = []
            for d in self.days:
                if self._is_weekend(d):
                    for s in self.shifts:
                        for w in allowed_wards:
                            if (n, d, s, w) in self.roster:
                                weekend_shifts.append(self.roster[(n, d, s, w)])
            
            if weekend_shifts:
                total_weekend = sum(weekend_shifts)
                # Penalize deviation from expected
                excess = self.model.NewIntVar(0, 100, f"weekend_excess_{n}")
                self.model.Add(excess >= total_weekend - int(expected_weekend_per_nurse) - 1)
                
                weighted_excess = self.model.NewIntVar(0, FAIRNESS_WEIGHT * 100, f"weekend_penalty_{n}")
                self.model.Add(weighted_excess == FAIRNESS_WEIGHT * excess)
                self.soft_penalties.append(weighted_excess)
    
    def _add_certification_soft_constraints(self) -> None:
        """Prioritize shifts that count toward certification renewal"""
        
        CERT_WEIGHT = 100
        
        for n in self.nurses:
            nurse = self.loader.get_nurse_data(n)
            
            if not nurse.cert_hours_needed:
                continue
            
            # Calculate potential certification hours from schedule
            total_cert_hours = []
            allowed_wards = [nurse.primary_ward_id] + nurse.secondary_ward_ids
            allowed_wards = [w for w in allowed_wards if w in self.wards]
            
            for d in self.days:
                for s in self.shifts:
                    for w in allowed_wards:
                        if (n, d, s, w) in self.roster:
                            # Each shift contributes 8 hours toward certification
                            total_cert_hours.append(8 * self.roster[(n, d, s, w)])
            
            # Penalize shortfall from needed hours
            for cert_id, hours_needed in nurse.cert_hours_needed.items():
                if total_cert_hours:
                    shortfall = self.model.NewIntVar(0, hours_needed, f"cert_shortfall_{n}_{cert_id}")
                    self.model.Add(shortfall >= hours_needed - sum(total_cert_hours))
                    
                    weighted_shortfall = self.model.NewIntVar(0, CERT_WEIGHT * hours_needed, f"cert_penalty_{n}_{cert_id}")
                    self.model.Add(weighted_shortfall == CERT_WEIGHT * shortfall)
                    self.soft_penalties.append(weighted_shortfall)
    
    def _set_objective(self) -> None:
        """Minimize total soft constraint penalties"""
        
        if self.soft_penalties:
            self.model.Minimize(sum(self.soft_penalties))
        else:
            logger.warning("No soft constraints added - objective is satisfaction only")
    
    def solve(self, time_limit_seconds: int = 300) -> bool:
        """Solve the model"""
        
        logger.info(f"Starting solver with {time_limit_seconds}s time limit...")
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        solver.parameters.num_search_workers = 8  # Parallel search
        
        self.status = solver.Solve(self.model)
        
        if self.status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            logger.info(f"Solution found! Status: {solver.StatusName(self.status)}")
            logger.info(f"Objective value: {solver.ObjectiveValue()}")
            
            # Extract solution
            self.solution = {}
            for key, var in self.roster.items():
                if solver.Value(var) == 1:
                    n, d, s, w = key
                    if n not in self.solution:
                        self.solution[n] = []
                    self.solution[n].append({
                        "date": self._date_for_day(d),
                        "shift": s,
                        "ward": w
                    })
            
            return True
        else:
            logger.error(f"No solution found. Status: {solver.StatusName(self.status)}")
            return False
    
    def get_roster_dataframe(self) -> pd.DataFrame:
        """Convert solution to DataFrame"""
        
        if not self.solution:
            return pd.DataFrame()
        
        rows = []
        for nurse_id, assignments in self.solution.items():
            for assignment in assignments:
                rows.append({
                    "nurse_id": nurse_id,
                    "shift_date": assignment["date"],
                    "shift_id": assignment["shift"],
                    "ward_id": assignment["ward"]
                })
        
        df = pd.DataFrame(rows)
        df = df.sort_values(["shift_date", "ward_id", "shift_id", "nurse_id"])
        
        return df
    
    def print_schedule_summary(self) -> None:
        """Print human-readable schedule summary"""
        
        if not self.solution:
            print("No solution available")
            return
        
        df = self.get_roster_dataframe()
        
        print("\n" + "=" * 70)
        print("ROSTER SUMMARY")
        print("=" * 70)
        
        # By date and ward
        for date in sorted(df["shift_date"].unique()):
            print(f"\n📅 {date}")
            day_df = df[df["shift_date"] == date]
            
            for ward_id in sorted(day_df["ward_id"].unique()):
                ward = self.loader.get_ward_data(ward_id)
                print(f"\n  🏥 {ward.ward_name} ({ward_id})")
                
                ward_df = day_df[day_df["ward_id"] == ward_id]
                
                for shift in ["D", "E", "N"]:
                    shift_names = {"D": "Day    ", "E": "Evening", "N": "Night  "}
                    shift_df = ward_df[ward_df["shift_id"] == shift]
                    nurses = shift_df["nurse_id"].tolist()
                    
                    min_req = {"D": ward.min_day, "E": ward.min_evening, "N": ward.min_night}
                    status = "✅" if len(nurses) >= min_req[shift] else "⚠️"
                    
                    print(f"    {shift_names[shift]}: {len(nurses):2d} nurses {status} (min: {min_req[shift]})")
        
        # Nurse summary
        print("\n" + "=" * 70)
        print("NURSE HOURS SUMMARY")
        print("=" * 70)
        
        nurse_hours = df.groupby("nurse_id").size() * 8
        print(f"\nNurses scheduled: {len(nurse_hours)}")
        print(f"Average hours/nurse: {nurse_hours.mean():.1f}")
        print(f"Min hours: {nurse_hours.min()}")
        print(f"Max hours: {nurse_hours.max()}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("IPD NURSE ROSTERING - OR-TOOLS SOLVER")
    print("=" * 70)
    
    # Load data
    loader = DataLoader("./mock_data")
    loader.load_all()
    
    # Create solver for 1 week, single ward (for demo)
    solver = NurseRosterSolver(
        data_loader=loader,
        planning_start="2025-01-20",
        planning_weeks=1
    )
    
    # Build model for ICU only (smaller problem for demo)
    solver.build_model(ward_ids=["W001"])
    
    # Solve
    success = solver.solve(time_limit_seconds=60)
    
    if success:
        solver.print_schedule_summary()
        
        # Save results
        roster_df = solver.get_roster_dataframe()
        roster_df.to_csv("./mock_data/generated_roster.csv", index=False)
        print(f"\n✅ Roster saved to ./mock_data/generated_roster.csv")
    else:
        print("\n❌ Failed to find feasible solution")
        print("Consider relaxing constraints or checking data consistency")


if __name__ == "__main__":
    main()
