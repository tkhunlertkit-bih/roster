# Databricks notebook source
# MAGIC %md
# MAGIC # IPD Nurse Rostering - Automated Schedule Generation
# MAGIC 
# MAGIC This notebook implements automated nurse roster generation using Google OR-Tools CP-SAT solver.
# MAGIC 
# MAGIC **Components:**
# MAGIC 1. Load data from Delta Lake tables
# MAGIC 2. Preprocess and validate constraints
# MAGIC 3. Run OR-Tools optimization
# MAGIC 4. Save results back to Delta
# MAGIC 5. Generate explanations using Azure AI Foundry
# MAGIC 
# MAGIC **Author:** AI Platform Team  
# MAGIC **Hospital:** Bumrungrad International Hospital

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & Configuration

# COMMAND ----------

# Install OR-Tools (run once per cluster)
%pip install ortools==9.8.3296 -q

# COMMAND ----------

# Configuration
CONFIG = {
    "catalog": "nursing",
    "schema": "roster",
    "planning_start": "2025-01-20",  # Monday
    "planning_weeks": 2,
    "solver_time_limit_seconds": 300,
    "wards_to_schedule": None,  # None = all wards
}

# Azure AI Foundry configuration (for explanations)
AZURE_AI_CONFIG = {
    "endpoint": dbutils.secrets.get(scope="azure-ai", key="endpoint"),
    "api_key": dbutils.secrets.get(scope="azure-ai", key="api-key"),
    "deployment": "gpt-4o",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Data from Delta Lake

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime, timedelta
import json
import pandas as pd

# Load dimension tables
dim_nurses = spark.table(f"{CONFIG['catalog']}.{CONFIG['schema']}.dim_nurses").filter("is_active = true")
dim_wards = spark.table(f"{CONFIG['catalog']}.{CONFIG['schema']}.dim_wards")
dim_shifts = spark.table(f"{CONFIG['catalog']}.{CONFIG['schema']}.dim_shifts")
dim_certifications = spark.table(f"{CONFIG['catalog']}.{CONFIG['schema']}.dim_certifications")
dim_rules = spark.table(f"{CONFIG['catalog']}.{CONFIG['schema']}.dim_scheduling_rules").filter("is_active = true")

# Load fact tables for planning period
planning_start = datetime.strptime(CONFIG["planning_start"], "%Y-%m-%d")
planning_end = planning_start + timedelta(weeks=CONFIG["planning_weeks"])

fact_nurse_certs = spark.table(f"{CONFIG['catalog']}.{CONFIG['schema']}.fact_nurse_certifications")
fact_unavailability = (
    spark.table(f"{CONFIG['catalog']}.{CONFIG['schema']}.fact_unavailability")
    .filter(f"status = 'APPROVED' AND end_date >= '{CONFIG['planning_start']}'")
)
fact_shift_requests = (
    spark.table(f"{CONFIG['catalog']}.{CONFIG['schema']}.fact_shift_requests")
    .filter(f"request_date >= '{CONFIG['planning_start']}' AND request_date < '{planning_end.strftime('%Y-%m-%d')}'")
)
fact_demand_forecast = (
    spark.table(f"{CONFIG['catalog']}.{CONFIG['schema']}.fact_demand_forecast")
    .filter(f"forecast_date >= '{CONFIG['planning_start']}' AND forecast_date < '{planning_end.strftime('%Y-%m-%d')}'")
)

print(f"Loaded {dim_nurses.count()} active nurses")
print(f"Loaded {dim_wards.count()} wards")
print(f"Planning period: {CONFIG['planning_start']} to {planning_end.strftime('%Y-%m-%d')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Convert to Pandas for OR-Tools

# COMMAND ----------

# Convert to pandas for local optimization
nurses_pdf = dim_nurses.toPandas()
wards_pdf = dim_wards.toPandas()
shifts_pdf = dim_shifts.toPandas()
certs_pdf = dim_certifications.toPandas()
nurse_certs_pdf = fact_nurse_certs.toPandas()
unavail_pdf = fact_unavailability.toPandas()
requests_pdf = fact_shift_requests.toPandas()
rules_pdf = dim_rules.toPandas()

# Display summary
display(nurses_pdf.groupby(['primary_ward_id', 'seniority']).size().reset_index(name='count'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. OR-Tools Solver Implementation

# COMMAND ----------

from ortools.sat.python import cp_model
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NurseRoster")


@dataclass
class NurseData:
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
    cert_hours_needed: Dict[str, int]


class NurseRosterSolver:
    """CP-SAT based nurse roster optimization"""
    
    def __init__(self, nurses_pdf, wards_pdf, nurse_certs_pdf, unavail_pdf,
                 planning_start: str, planning_weeks: int):
        self.nurses_pdf = nurses_pdf
        self.wards_pdf = wards_pdf
        self.nurse_certs_pdf = nurse_certs_pdf
        self.unavail_pdf = unavail_pdf
        
        self.start_date = datetime.strptime(planning_start, "%Y-%m-%d")
        self.num_days = planning_weeks * 7
        
        self.model = cp_model.CpModel()
        self.roster = {}
        self.soft_penalties = []
        
        self.nurses = []
        self.wards = []
        self.shifts = ["D", "E", "N"]
        self.days = list(range(self.num_days))
        
        self.solution = None
        
    def _date_for_day(self, day_index: int) -> str:
        return (self.start_date + timedelta(days=day_index)).strftime("%Y-%m-%d")
    
    def _is_weekend(self, day_index: int) -> bool:
        return (self.start_date + timedelta(days=day_index)).weekday() >= 5
    
    def get_nurse_data(self, nurse_id: str) -> NurseData:
        nurse = self.nurses_pdf[self.nurses_pdf["nurse_id"] == nurse_id].iloc[0]
        
        # Get certifications
        nurse_certs = self.nurse_certs_pdf[
            (self.nurse_certs_pdf["nurse_id"] == nurse_id) & 
            (self.nurse_certs_pdf["is_expired"] == False)
        ]
        certs = nurse_certs["cert_id"].tolist()
        
        # Get cert hours needed
        cert_hours = {}
        for _, nc in nurse_certs.iterrows():
            if nc["hours_remaining"] > 0:
                cert_hours[nc["cert_id"]] = int(nc["hours_remaining"])
        
        # Get unavailable dates
        unavail = self.unavail_pdf[
            (self.unavail_pdf["nurse_id"] == nurse_id) & 
            (self.unavail_pdf["status"] == "APPROVED")
        ]
        unavail_dates = []
        for _, u in unavail.iterrows():
            start = datetime.strptime(str(u["start_date"])[:10], "%Y-%m-%d")
            end = datetime.strptime(str(u["end_date"])[:10], "%Y-%m-%d")
            current = start
            while current <= end:
                unavail_dates.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
        
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
    
    def get_ward_data(self, ward_id: str):
        ward = self.wards_pdf[self.wards_pdf["ward_id"] == ward_id].iloc[0]
        try:
            required_certs = json.loads(ward["requires_certification"])
        except:
            required_certs = []
        return {
            "ward_id": ward_id,
            "ward_name": ward["ward_name"],
            "min_day": int(ward["min_nurses_day"]),
            "min_evening": int(ward["min_nurses_evening"]),
            "min_night": int(ward["min_nurses_night"]),
            "required_certs": required_certs,
            "acuity": int(ward["acuity_level"])
        }
    
    def get_eligible_nurses_for_ward(self, ward_id: str) -> List[str]:
        ward = self.get_ward_data(ward_id)
        ward_nurses = self.nurses_pdf[
            self.nurses_pdf["primary_ward_id"] == ward_id
        ]["nurse_id"].tolist()
        
        if not ward["required_certs"]:
            return ward_nurses
        
        eligible = []
        for nurse_id in ward_nurses:
            nurse = self.get_nurse_data(nurse_id)
            if all(cert in nurse.certifications for cert in ward["required_certs"]):
                eligible.append(nurse_id)
        
        return eligible
    
    def build_model(self, ward_ids: Optional[List[str]] = None):
        logger.info("Building CP-SAT model...")
        
        if ward_ids:
            self.wards = ward_ids
        else:
            self.wards = self.wards_pdf["ward_id"].tolist()
        
        self.nurses = []
        for ward_id in self.wards:
            ward_nurses = self.get_eligible_nurses_for_ward(ward_id)
            self.nurses.extend(ward_nurses)
        self.nurses = list(set(self.nurses))
        
        logger.info(f"Building model for {len(self.nurses)} nurses, {len(self.wards)} wards, {self.num_days} days")
        
        # Create variables
        for n in self.nurses:
            nurse = self.get_nurse_data(n)
            allowed_wards = [nurse.primary_ward_id] + nurse.secondary_ward_ids
            
            for d in self.days:
                for s in self.shifts:
                    for w in self.wards:
                        if w in allowed_wards:
                            self.roster[(n, d, s, w)] = self.model.NewBoolVar(f"r_{n}_{d}_{s}_{w}")
        
        # Add constraints (simplified version)
        self._add_one_shift_per_day()
        self._add_weekly_hours_constraint()
        self._add_min_coverage_constraints()
        self._add_unavailability_constraints()
        
        # Set objective
        if self.soft_penalties:
            self.model.Minimize(sum(self.soft_penalties))
        
        logger.info(f"Model built with {len(self.roster)} variables")
    
    def _add_one_shift_per_day(self):
        for n in self.nurses:
            nurse = self.get_nurse_data(n)
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
    
    def _add_weekly_hours_constraint(self):
        for n in self.nurses:
            nurse = self.get_nurse_data(n)
            allowed_wards = [nurse.primary_ward_id] + nurse.secondary_ward_ids
            allowed_wards = [w for w in allowed_wards if w in self.wards]
            target_shifts = nurse.contracted_hours // 8
            
            for week in range(self.num_days // 7):
                week_start = week * 7
                week_end = min((week + 1) * 7, self.num_days)
                
                available_days = sum(
                    1 for d in range(week_start, week_end)
                    if self._date_for_day(d) not in nurse.unavailable_dates
                )
                
                total_shifts = []
                for d in range(week_start, week_end):
                    for s in self.shifts:
                        for w in allowed_wards:
                            if (n, d, s, w) in self.roster:
                                total_shifts.append(self.roster[(n, d, s, w)])
                
                if total_shifts and available_days > 0:
                    max_possible = min(available_days, target_shifts + 1)
                    min_required = max(0, min(available_days, target_shifts - 2))
                    self.model.Add(sum(total_shifts) >= min_required)
                    self.model.Add(sum(total_shifts) <= max_possible)
    
    def _add_min_coverage_constraints(self):
        COVERAGE_PENALTY = 500
        
        for w in self.wards:
            ward = self.get_ward_data(w)
            min_staff = {"D": ward["min_day"], "E": ward["min_evening"], "N": ward["min_night"]}
            
            for d in self.days:
                for s in self.shifts:
                    assigned = [self.roster[(n, d, s, w)] for n in self.nurses if (n, d, s, w) in self.roster]
                    
                    if assigned:
                        total_assigned = sum(assigned)
                        shortfall = self.model.NewIntVar(0, min_staff[s], f"sf_{w}_{d}_{s}")
                        self.model.Add(shortfall >= min_staff[s] - total_assigned)
                        
                        penalty = self.model.NewIntVar(0, COVERAGE_PENALTY * min_staff[s], f"cp_{w}_{d}_{s}")
                        self.model.Add(penalty == COVERAGE_PENALTY * shortfall)
                        self.soft_penalties.append(penalty)
                        
                        self.model.Add(total_assigned >= 1)
    
    def _add_unavailability_constraints(self):
        for n in self.nurses:
            nurse = self.get_nurse_data(n)
            
            for d in self.days:
                date_str = self._date_for_day(d)
                if date_str in nurse.unavailable_dates:
                    for s in self.shifts:
                        for w in self.wards:
                            if (n, d, s, w) in self.roster:
                                self.model.Add(self.roster[(n, d, s, w)] == 0)
    
    def solve(self, time_limit_seconds: int = 300) -> bool:
        logger.info(f"Starting solver with {time_limit_seconds}s time limit...")
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        solver.parameters.num_search_workers = 8
        
        status = solver.Solve(self.model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            logger.info(f"Solution found! Status: {solver.StatusName(status)}")
            logger.info(f"Objective value: {solver.ObjectiveValue()}")
            
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
            logger.error(f"No solution found. Status: {solver.StatusName(status)}")
            return False
    
    def get_roster_dataframe(self) -> pd.DataFrame:
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
        return df.sort_values(["shift_date", "ward_id", "shift_id", "nurse_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Run Optimization

# COMMAND ----------

# Create and run solver
solver = NurseRosterSolver(
    nurses_pdf=nurses_pdf,
    wards_pdf=wards_pdf,
    nurse_certs_pdf=nurse_certs_pdf,
    unavail_pdf=unavail_pdf,
    planning_start=CONFIG["planning_start"],
    planning_weeks=CONFIG["planning_weeks"]
)

# Build model (for specific wards or all)
solver.build_model(ward_ids=CONFIG["wards_to_schedule"])

# Solve
success = solver.solve(time_limit_seconds=CONFIG["solver_time_limit_seconds"])

if success:
    roster_pdf = solver.get_roster_dataframe()
    print(f"Generated roster with {len(roster_pdf)} assignments")
    display(roster_pdf.head(20))
else:
    raise Exception("Failed to find feasible roster solution")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Results to Delta Lake

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import uuid

# Add metadata
roster_pdf["roster_id"] = [str(uuid.uuid4()) for _ in range(len(roster_pdf))]
roster_pdf["planning_run_id"] = str(uuid.uuid4())
roster_pdf["planning_start"] = CONFIG["planning_start"]
roster_pdf["created_at"] = datetime.now()
roster_pdf["status"] = "DRAFT"

# Convert to Spark DataFrame
roster_sdf = spark.createDataFrame(roster_pdf)

# Write to Delta
roster_sdf.write.format("delta").mode("append").saveAsTable(
    f"{CONFIG['catalog']}.{CONFIG['schema']}.fact_generated_roster"
)

print(f"✅ Saved {len(roster_pdf)} roster entries to Delta Lake")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Generate Coverage Analysis

# COMMAND ----------

# Coverage analysis
coverage_analysis = (
    roster_sdf
    .groupBy("shift_date", "ward_id", "shift_id")
    .count()
    .withColumnRenamed("count", "assigned_nurses")
)

# Join with ward requirements
coverage_with_req = (
    coverage_analysis
    .join(dim_wards.select("ward_id", "ward_name", "min_nurses_day", "min_nurses_evening", "min_nurses_night"), "ward_id")
)

display(coverage_with_req.orderBy("shift_date", "ward_id", "shift_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Azure AI Foundry Integration (Optional)
# MAGIC 
# MAGIC Generate natural language explanations for the schedule.

# COMMAND ----------

import requests

def explain_schedule_with_ai(roster_summary: str, constraints: str) -> str:
    """Use Azure OpenAI to explain schedule decisions"""
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_AI_CONFIG["api_key"]
    }
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": """You are a helpful assistant that explains nurse scheduling decisions 
                to head nurses in clear, professional language. Focus on:
                1. Why specific assignments were made
                2. How constraints were balanced
                3. Any coverage gaps and recommendations"""
            },
            {
                "role": "user", 
                "content": f"""Please explain this schedule to the Head Nurse:

Schedule Summary:
{roster_summary}

Constraints Applied:
{constraints}

Provide a clear explanation of the schedule quality and any areas needing attention."""
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.3
    }
    
    response = requests.post(
        f"{AZURE_AI_CONFIG['endpoint']}/openai/deployments/{AZURE_AI_CONFIG['deployment']}/chat/completions?api-version=2024-02-15-preview",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# Example usage (uncomment to run)
# schedule_summary = roster_sdf.toPandas().to_string()
# constraints_summary = rules_pdf[rules_pdf["constraint_type"] == "HARD"]["rule_name"].tolist()
# explanation = explain_schedule_with_ai(schedule_summary[:2000], str(constraints_summary))
# print(explanation)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary Dashboard

# COMMAND ----------

# Final summary
print("=" * 70)
print("ROSTER GENERATION COMPLETE")
print("=" * 70)
print(f"Planning Period: {CONFIG['planning_start']} to {planning_end.strftime('%Y-%m-%d')}")
print(f"Wards Scheduled: {len(solver.wards)}")
print(f"Nurses Scheduled: {len(solver.solution) if solver.solution else 0}")
print(f"Total Assignments: {len(roster_pdf)}")
print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
print("=" * 70)
