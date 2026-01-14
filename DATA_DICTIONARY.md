# IPD Nurse Rostering - Data Dictionary

## Overview

This document describes the data model for the IPD Nurse Rostering optimization system designed for **Bumrungrad International Hospital**.

### Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   dim_wards     │       │   dim_shifts    │       │dim_certifications│
│                 │       │                 │       │                 │
│ ward_id (PK)    │       │ shift_id (PK)   │       │ cert_id (PK)    │
│ ward_name       │       │ shift_name      │       │ cert_name       │
│ min_nurses_*    │       │ duration_hours  │       │ validity_years  │
│ requires_cert[] │       │ cert_weight     │       │ renewal_hours   │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         │    ┌────────────────────┼─────────────────────────┤
         │    │                    │                         │
         │    │    ┌───────────────┴───────────────┐         │
         ▼    ▼    ▼                               │         │
┌─────────────────────────────────────┐           │         │
│            dim_nurses               │           │         │
│                                     │           │         │
│  nurse_id (PK)                      │           │         │
│  primary_ward_id (FK) ─────────────►│           │         │
│  contracted_hours_week              │           │         │
│  seniority, skill_level             │           │         │
│  preferences (night/day)            │           │         │
└──────────────┬──────────────────────┘           │         │
               │                                   │         │
               │                                   │         │
    ┌──────────┴──────────┬───────────────────────┤         │
    │                     │                       │         │
    ▼                     ▼                       ▼         ▼
┌───────────────┐  ┌──────────────┐  ┌─────────────────────────────┐
│fact_unavail   │  │fact_shift_req│  │ fact_nurse_certifications   │
│               │  │              │  │                             │
│unavail_id(PK) │  │request_id(PK)│  │ nurse_cert_id (PK)          │
│nurse_id (FK)  │  │nurse_id (FK) │  │ nurse_id (FK)               │
│unavail_type   │  │request_type  │  │ cert_id (FK)                │
│start/end_date │  │request_date  │  │ practice_hours_ytd          │
│status         │  │preferred_shft│  │ hours_remaining             │
└───────────────┘  └──────────────┘  │ expiry_date                 │
                                     └─────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  fact_historical_roster │
              │                         │
              │  roster_id (PK)         │
              │  nurse_id (FK)          │
              │  ward_id (FK)           │
              │  shift_id (FK)          │
              │  shift_date             │
              │  actual_hours           │
              └─────────────────────────┘
```

---

## Dimension Tables

### 1. `dim_wards` - Hospital Wards/Units

Master data for hospital wards where nurses are assigned.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `ward_id` | STRING | Primary key | W001 |
| `ward_name` | STRING | Display name | ICU |
| `ward_type` | STRING | Category: CRITICAL, MEDICAL, SURGICAL, MATERNITY, PEDIATRIC | CRITICAL |
| `floor` | INT | Physical floor number | 7 |
| `bed_count` | INT | Number of beds | 20 |
| `min_nurses_day` | INT | Minimum RNs for day shift | 8 |
| `min_nurses_evening` | INT | Minimum RNs for evening shift | 6 |
| `min_nurses_night` | INT | Minimum RNs for night shift | 5 |
| `requires_certification` | JSON | Array of required cert_ids | ["ICU", "ACLS"] |
| `acuity_level` | INT | Patient acuity (1-5) | 5 |

**Usage in OR-Tools**: Hard constraints for minimum coverage per shift.

---

### 2. `dim_shifts` - Shift Definitions

Defines available shift types and their properties.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `shift_id` | STRING | Primary key | D, E, N, D12, OFF |
| `shift_name` | STRING | Display name | Day, Evening, Night |
| `start_time` | STRING | Shift start (HH:MM) | 07:00 |
| `end_time` | STRING | Shift end (HH:MM) | 15:00 |
| `duration_hours` | INT | Shift length | 8 |
| `shift_type` | STRING | REGULAR, EXTENDED, OFF, LEAVE, TRAINING | REGULAR |
| `overtime_multiplier` | FLOAT | Pay multiplier | 1.0 |
| `night_differential` | FLOAT | Additional pay % for night | 0.15 |
| `counts_for_certification` | BOOL | Counts toward cert practice hours | true |
| `certification_weight` | FLOAT | Hours multiplier for cert tracking | 1.2 |

**Usage in OR-Tools**: Decision variable domain; soft constraint weights for certification.

---

### 3. `dim_certifications` - Certification Types

Master data for nursing certifications and licenses.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `cert_id` | STRING | Primary key | ICU, ACLS, BLS |
| `cert_name` | STRING | Full name | Advanced Cardiac Life Support |
| `cert_type` | STRING | LICENSE, SPECIALTY, SKILL | SKILL |
| `issuing_body` | STRING | Certifying organization | American Heart Association |
| `validity_years` | INT | Years before renewal | 2 |
| `renewal_requires_hours` | INT | Practice hours for renewal | 40 |
| `renewal_requires_exam` | BOOL | Exam required for renewal | true |
| `is_mandatory` | BOOL | Required for all nurses | false |

**Usage in OR-Tools**: Hard constraint (ward eligibility); soft constraint (practice hours).

---

### 4. `dim_nurses` - Nurse Master Data

Complete nurse profiles including preferences and constraints.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `nurse_id` | STRING | Primary key | N0001 |
| `employee_id` | STRING | HR system ID | EMP010001 |
| `first_name` | STRING | Given name | Somchai |
| `last_name` | STRING | Family name | Wongsakorn |
| `years_experience` | INT | Years as RN | 8 |
| `seniority` | STRING | JUNIOR, MID, SENIOR | MID |
| `position` | STRING | NEW_GRAD, RN, SENIOR_RN, HEAD_NURSE | RN |
| `employment_type` | STRING | FULL_TIME, PART_TIME, CONTRACT | FULL_TIME |
| `contracted_hours_week` | INT | Weekly hours commitment | 40 |
| `primary_ward_id` | STRING | FK to dim_wards | W001 |
| `secondary_ward_ids` | JSON | Wards nurse can float to | ["W002"] |
| `skill_level` | INT | Competency (1-5) | 4 |
| `can_charge_nurse` | BOOL | Can serve as charge | true |
| `can_preceptor` | BOOL | Can train new nurses | true |
| `prefers_day_shift` | BOOL | Day shift preference | true |
| `prefers_night_shift` | BOOL | Night shift preference | false |
| `max_consecutive_days` | INT | Personal max consecutive | 5 |
| `max_night_shifts_week` | INT | Personal max nights/week | 3 |
| `max_weekend_shifts_month` | INT | Personal max weekends | 4 |
| `is_active` | BOOL | Currently employed | true |

**Usage in OR-Tools**: Defines individual constraint bounds and preference weights.

---

### 5. `dim_scheduling_rules` - Rule Engine

Configurable constraints for the optimization model.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `rule_id` | STRING | Primary key | R001 |
| `rule_name` | STRING | Display name | Weekly Hours Limit |
| `rule_type` | STRING | Category: HOURS, REST, COVERAGE, etc. | HOURS |
| `constraint_type` | STRING | HARD or SOFT | HARD |
| `priority` | INT | 1=highest, 3=lowest | 1 |
| `scope` | STRING | GLOBAL, WARD, INDIVIDUAL | GLOBAL |
| `applies_to_ward` | STRING | Specific ward or ALL | ALL |
| `applies_to_position` | STRING | Specific position or NULL | NULL |
| `parameters` | JSON | Rule-specific parameters | {"max_hours_week": 40} |
| `is_active` | BOOL | Rule enabled | true |
| `effective_from` | DATE | Start validity | 2024-01-01 |
| `effective_to` | DATE | End validity (NULL=forever) | NULL |

**Rule Types:**
- `HOURS`: Weekly/daily hour limits
- `REST`: Minimum rest between shifts
- `CONSECUTIVE`: Max consecutive working days
- `COVERAGE`: Minimum staffing requirements
- `SEQUENCE`: Forbidden shift sequences (e.g., E→N)
- `QUALIFICATION`: Certification requirements
- `CERTIFICATION`: Practice hour tracking
- `PREFERENCE`: Nurse preferences
- `FAIRNESS`: Equal distribution of undesirable shifts

---

### 6. `dim_special_events` - Events Affecting Scheduling

External events that impact staffing needs.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `event_id` | STRING | Primary key | EVT001 |
| `event_name` | STRING | Event description | JCI Accreditation Visit |
| `event_date` | DATE | Date of event | 2025-01-27 |
| `event_type` | STRING | INSPECTION, HOLIDAY, TRAINING | INSPECTION |
| `affected_wards` | JSON | Ward IDs or ["ALL"] | ["ALL"] |
| `staffing_impact` | STRING | INCREASE, DECREASE, HIGH_LEAVE_RISK | INCREASE |
| `impact_percentage` | INT | % change in requirements | 20 |

---

## Fact Tables

### 7. `fact_nurse_certifications` - Nurse Certification Holdings

Junction table tracking which certifications each nurse holds.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `nurse_cert_id` | STRING | Primary key (UUID) | uuid-xxx |
| `nurse_id` | STRING | FK to dim_nurses | N0001 |
| `cert_id` | STRING | FK to dim_certifications | ICU |
| `obtained_date` | DATE | Date certified | 2023-06-15 |
| `expiry_date` | DATE | Certification expiry | 2025-06-15 |
| `is_expired` | BOOL | Currently expired | false |
| `practice_hours_ytd` | INT | Hours practiced this year | 28 |
| `practice_hours_required` | INT | Hours needed for renewal | 40 |
| `hours_remaining` | INT | Gap to meet requirement | 12 |
| `renewal_status` | STRING | CURRENT, EXPIRING_SOON, EXPIRED | CURRENT |

**Usage in OR-Tools**: 
- `is_expired=false` + required certs → Hard constraint (eligibility)
- `hours_remaining > 0` → Soft constraint (prioritize cert-qualifying shifts)

---

### 8. `fact_unavailability` - Leave and Unavailability

Pre-existing commitments that block scheduling.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `unavail_id` | STRING | Primary key (UUID) | uuid-xxx |
| `nurse_id` | STRING | FK to dim_nurses | N0001 |
| `unavail_type` | STRING | ANNUAL_LEAVE, SICK_LEAVE, TRAINING, RECURRING | ANNUAL_LEAVE |
| `start_date` | DATE | First unavailable date | 2025-01-22 |
| `end_date` | DATE | Last unavailable date | 2025-01-24 |
| `is_recurring` | BOOL | Repeats weekly | false |
| `recurrence_pattern` | JSON | Pattern if recurring | {"day_of_week": 0} |
| `status` | STRING | PENDING, APPROVED, DENIED | APPROVED |

**Usage in OR-Tools**: `status=APPROVED` → Hard constraint (nurse unavailable).

---

### 9. `fact_shift_requests` - Nurse Shift Preferences

Specific requests from nurses for particular dates.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `request_id` | STRING | Primary key (UUID) | uuid-xxx |
| `nurse_id` | STRING | FK to dim_nurses | N0001 |
| `request_type` | STRING | REQUEST_ON, REQUEST_OFF, SWAP_REQUEST | REQUEST_OFF |
| `request_date` | DATE | Date of request | 2025-01-25 |
| `preferred_shift` | STRING | Shift ID if requesting on | D |
| `swap_with_nurse_id` | STRING | Partner for swap | N0042 |
| `priority` | STRING | LOW, MEDIUM, HIGH | HIGH |
| `status` | STRING | PENDING, APPROVED, DENIED | PENDING |

**Usage in OR-Tools**: 
- `APPROVED` → Hard constraint
- `PENDING` + priority → Soft constraint weight

---

### 10. `fact_historical_roster` - Past Schedules

Historical roster data for analytics and fairness calculations.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `roster_id` | STRING | Primary key (UUID) | uuid-xxx |
| `nurse_id` | STRING | FK to dim_nurses | N0001 |
| `ward_id` | STRING | FK to dim_wards | W001 |
| `shift_date` | DATE | Date worked | 2025-01-15 |
| `shift_id` | STRING | FK to dim_shifts | D |
| `actual_hours` | FLOAT | Hours worked | 8.0 |
| `is_overtime` | BOOL | OT flag | false |
| `is_charge_nurse` | BOOL | Was charge nurse | true |
| `status` | STRING | COMPLETED, NO_SHOW, SICK_CALL | COMPLETED |

**Usage in OR-Tools**: 
- Calculate cumulative weekend/night shifts for fairness
- Identify patterns for soft preference learning

---

### 11. `fact_demand_forecast` - Expected Patient Census

Predicted staffing needs based on census forecasts.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `forecast_id` | STRING | Primary key (UUID) | uuid-xxx |
| `ward_id` | STRING | FK to dim_wards | W001 |
| `forecast_date` | DATE | Date forecasted | 2025-01-22 |
| `expected_census` | INT | Expected patients | 18 |
| `expected_occupancy_pct` | FLOAT | Bed occupancy % | 90.0 |
| `expected_avg_acuity` | FLOAT | Average acuity (1-5) | 4.2 |
| `confidence_level` | STRING | HIGH, MEDIUM, LOW | HIGH |

**Usage in OR-Tools**: Dynamically adjust `min_nurses_*` based on census/acuity.

---

## Data Volume Summary

| Table | Rows | Update Frequency |
|-------|------|------------------|
| dim_wards | 8 | Rarely |
| dim_shifts | 9 | Rarely |
| dim_certifications | 10 | Rarely |
| dim_nurses | 120 | Weekly |
| dim_scheduling_rules | 18 | Monthly |
| dim_special_events | 3 | As needed |
| fact_nurse_certifications | 528 | Monthly |
| fact_unavailability | 83 | Daily |
| fact_shift_requests | 97 | Daily |
| fact_historical_roster | 6,840 | Daily (append) |
| fact_demand_forecast | 224 | Daily |

---

## Databricks Delta Table Schema (Recommended)

```sql
-- Example: Create dim_nurses as Delta table
CREATE TABLE IF NOT EXISTS nursing.dim_nurses (
    nurse_id STRING NOT NULL,
    employee_id STRING,
    first_name STRING,
    last_name STRING,
    years_experience INT,
    seniority STRING,
    position STRING,
    employment_type STRING,
    contracted_hours_week INT,
    primary_ward_id STRING,
    secondary_ward_ids STRING,  -- JSON array
    skill_level INT,
    can_charge_nurse BOOLEAN,
    can_preceptor BOOLEAN,
    prefers_day_shift BOOLEAN,
    prefers_night_shift BOOLEAN,
    max_consecutive_days INT,
    max_night_shifts_week INT,
    max_weekend_shifts_month INT,
    is_active BOOLEAN,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)
USING DELTA
PARTITIONED BY (primary_ward_id)
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
);
```

---

## Next Steps

1. **Load to Databricks**: Use the provided CSVs to create Delta tables
2. **Build Preprocessing Pipeline**: Transform data into OR-Tools input format
3. **Implement Solver**: Use the sample `or_tools_solver.py` as starting point
4. **Integrate Azure AI**: Add natural language interface for schedule explanations
