"""
IPD Nurse Rostering - Mock Data Generator
==========================================
Generates realistic mock data for nurse scheduling optimization
Designed for Bumrungrad-style hospital operations

Author: AI Assistant
Use Case: OR-Tools based nurse rostering with Databricks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import json
import random
from typing import List, Dict, Any
import uuid

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "num_nurses": 120,
    "num_wards": 8,
    "roster_weeks": 4,  # Generate 4 weeks of planning horizon
    "start_date": "2025-01-20",  # Monday
    "hospital_name": "Bumrungrad International Hospital",
}

# Thai names for realistic data
THAI_FIRST_NAMES = [
    "Somchai", "Somsak", "Sompong", "Prasit", "Prawit", "Prayuth", "Anong", 
    "Aranya", "Busaba", "Chanida", "Duangjai", "Fongchan", "Jiraporn", 
    "Kannika", "Lalita", "Malai", "Nanthida", "Orathai", "Parichat", 
    "Ratana", "Siriporn", "Thidarat", "Urai", "Vanida", "Warunee",
    "Yuphin", "Apinya", "Benjamas", "Chutima", "Darunee", "Hathaichanok",
    "Isaree", "Jutamas", "Kanya", "Laddawan", "Manee", "Narumol",
    "Orawan", "Pensri", "Ratchanee", "Supatra", "Thaniya", "Usanee",
    "Wilaiwan", "Yaowapa", "Anchalee", "Boonsri", "Chamaiporn", "Duangkamol"
]

THAI_LAST_NAMES = [
    "Wongsakorn", "Srisawat", "Thongchai", "Prasertsin", "Rattanakorn",
    "Sukhumvit", "Chaiyaporn", "Pattanapong", "Sirichote", "Boonmee",
    "Charoenrat", "Duangmanee", "Ekachai", "Fuangfoo", "Gongsak",
    "Hongthai", "Intharachai", "Jantawong", "Kittisak", "Laohavisit",
    "Mongkolsiri", "Nuanphan", "Ongart", "Petcharat", "Rungruang",
    "Somboon", "Thanakit", "Udomsak", "Vejjajiva", "Wattanasiri"
]

# =============================================================================
# 1. WARDS / UNITS
# =============================================================================

def generate_wards() -> pd.DataFrame:
    """Generate hospital wards/units master data"""
    
    wards = [
        {"ward_id": "W001", "ward_name": "ICU", "ward_type": "CRITICAL", 
         "floor": 7, "bed_count": 20, "min_nurses_day": 8, "min_nurses_evening": 6, 
         "min_nurses_night": 5, "requires_certification": ["ICU", "ACLS"], "acuity_level": 5},
        
        {"ward_id": "W002", "ward_name": "CCU", "ward_type": "CRITICAL",
         "floor": 7, "bed_count": 16, "min_nurses_day": 6, "min_nurses_evening": 5,
         "min_nurses_night": 4, "requires_certification": ["CCU", "ACLS"], "acuity_level": 5},
        
        {"ward_id": "W003", "ward_name": "Medical Ward 5A", "ward_type": "MEDICAL",
         "floor": 5, "bed_count": 32, "min_nurses_day": 6, "min_nurses_evening": 5,
         "min_nurses_night": 4, "requires_certification": [], "acuity_level": 3},
        
        {"ward_id": "W004", "ward_name": "Medical Ward 5B", "ward_type": "MEDICAL",
         "floor": 5, "bed_count": 32, "min_nurses_day": 6, "min_nurses_evening": 5,
         "min_nurses_night": 4, "requires_certification": [], "acuity_level": 3},
        
        {"ward_id": "W005", "ward_name": "Surgical Ward 6A", "ward_type": "SURGICAL",
         "floor": 6, "bed_count": 28, "min_nurses_day": 7, "min_nurses_evening": 5,
         "min_nurses_night": 4, "requires_certification": ["POST_OP"], "acuity_level": 4},
        
        {"ward_id": "W006", "ward_name": "Surgical Ward 6B", "ward_type": "SURGICAL",
         "floor": 6, "bed_count": 28, "min_nurses_day": 7, "min_nurses_evening": 5,
         "min_nurses_night": 4, "requires_certification": ["POST_OP"], "acuity_level": 4},
        
        {"ward_id": "W007", "ward_name": "Maternity Ward", "ward_type": "MATERNITY",
         "floor": 4, "bed_count": 24, "min_nurses_day": 6, "min_nurses_evening": 5,
         "min_nurses_night": 4, "requires_certification": ["OB_NURS"], "acuity_level": 3},
        
        {"ward_id": "W008", "ward_name": "Pediatric Ward", "ward_type": "PEDIATRIC",
         "floor": 4, "bed_count": 20, "min_nurses_day": 5, "min_nurses_evening": 4,
         "min_nurses_night": 3, "requires_certification": ["PEDS"], "acuity_level": 3},
    ]
    
    df = pd.DataFrame(wards)
    df["requires_certification"] = df["requires_certification"].apply(json.dumps)
    df["created_at"] = datetime.now()
    df["updated_at"] = datetime.now()
    
    return df


# =============================================================================
# 2. SHIFTS
# =============================================================================

def generate_shifts() -> pd.DataFrame:
    """Generate shift definitions"""
    
    shifts = [
        {"shift_id": "D", "shift_name": "Day", "start_time": "07:00", 
         "end_time": "15:00", "duration_hours": 8, "shift_type": "REGULAR",
         "overtime_multiplier": 1.0, "night_differential": 0.0,
         "counts_for_certification": True, "certification_weight": 1.0},
        
        {"shift_id": "E", "shift_name": "Evening", "start_time": "15:00",
         "end_time": "23:00", "duration_hours": 8, "shift_type": "REGULAR",
         "overtime_multiplier": 1.0, "night_differential": 0.0,
         "counts_for_certification": True, "certification_weight": 1.0},
        
        {"shift_id": "N", "shift_name": "Night", "start_time": "23:00",
         "end_time": "07:00", "duration_hours": 8, "shift_type": "REGULAR",
         "overtime_multiplier": 1.0, "night_differential": 0.15,
         "counts_for_certification": True, "certification_weight": 1.2},
        
        {"shift_id": "D12", "shift_name": "Day 12hr", "start_time": "07:00",
         "end_time": "19:00", "duration_hours": 12, "shift_type": "EXTENDED",
         "overtime_multiplier": 1.0, "night_differential": 0.0,
         "counts_for_certification": True, "certification_weight": 1.5},
        
        {"shift_id": "N12", "shift_name": "Night 12hr", "start_time": "19:00",
         "end_time": "07:00", "duration_hours": 12, "shift_type": "EXTENDED",
         "overtime_multiplier": 1.0, "night_differential": 0.15,
         "counts_for_certification": True, "certification_weight": 1.8},
        
        {"shift_id": "OFF", "shift_name": "Day Off", "start_time": None,
         "end_time": None, "duration_hours": 0, "shift_type": "OFF",
         "overtime_multiplier": 0.0, "night_differential": 0.0,
         "counts_for_certification": False, "certification_weight": 0.0},
        
        {"shift_id": "AL", "shift_name": "Annual Leave", "start_time": None,
         "end_time": None, "duration_hours": 0, "shift_type": "LEAVE",
         "overtime_multiplier": 0.0, "night_differential": 0.0,
         "counts_for_certification": False, "certification_weight": 0.0},
        
        {"shift_id": "SL", "shift_name": "Sick Leave", "start_time": None,
         "end_time": None, "duration_hours": 0, "shift_type": "LEAVE",
         "overtime_multiplier": 0.0, "night_differential": 0.0,
         "counts_for_certification": False, "certification_weight": 0.0},
        
        {"shift_id": "TR", "shift_name": "Training", "start_time": "08:00",
         "end_time": "17:00", "duration_hours": 8, "shift_type": "TRAINING",
         "overtime_multiplier": 1.0, "night_differential": 0.0,
         "counts_for_certification": True, "certification_weight": 0.5},
    ]
    
    return pd.DataFrame(shifts)


# =============================================================================
# 3. CERTIFICATIONS
# =============================================================================

def generate_certifications() -> pd.DataFrame:
    """Generate certification types and requirements"""
    
    certifications = [
        {"cert_id": "RN", "cert_name": "Registered Nurse", "cert_type": "LICENSE",
         "issuing_body": "Thailand Nursing Council", "validity_years": 5,
         "renewal_requires_hours": 50, "renewal_requires_exam": True,
         "is_mandatory": True, "description": "Basic nursing license"},
        
        {"cert_id": "ICU", "cert_name": "ICU Specialty", "cert_type": "SPECIALTY",
         "issuing_body": "Thai Critical Care Nursing Society", "validity_years": 2,
         "renewal_requires_hours": 40, "renewal_requires_exam": False,
         "is_mandatory": False, "description": "Intensive Care Unit certification"},
        
        {"cert_id": "CCU", "cert_name": "Cardiac Care", "cert_type": "SPECIALTY",
         "issuing_body": "Thai Heart Association", "validity_years": 2,
         "renewal_requires_hours": 40, "renewal_requires_exam": False,
         "is_mandatory": False, "description": "Cardiac Care Unit certification"},
        
        {"cert_id": "ACLS", "cert_name": "Advanced Cardiac Life Support", "cert_type": "SKILL",
         "issuing_body": "American Heart Association", "validity_years": 2,
         "renewal_requires_hours": 16, "renewal_requires_exam": True,
         "is_mandatory": False, "description": "Emergency cardiac care"},
        
        {"cert_id": "BLS", "cert_name": "Basic Life Support", "cert_type": "SKILL",
         "issuing_body": "American Heart Association", "validity_years": 2,
         "renewal_requires_hours": 8, "renewal_requires_exam": True,
         "is_mandatory": True, "description": "Basic emergency response"},
        
        {"cert_id": "PEDS", "cert_name": "Pediatric Nursing", "cert_type": "SPECIALTY",
         "issuing_body": "Thai Pediatric Nursing Society", "validity_years": 3,
         "renewal_requires_hours": 30, "renewal_requires_exam": False,
         "is_mandatory": False, "description": "Pediatric care specialty"},
        
        {"cert_id": "OB_NURS", "cert_name": "Obstetric Nursing", "cert_type": "SPECIALTY",
         "issuing_body": "Thai OB-GYN Nursing Society", "validity_years": 3,
         "renewal_requires_hours": 30, "renewal_requires_exam": False,
         "is_mandatory": False, "description": "Obstetric and maternity care"},
        
        {"cert_id": "POST_OP", "cert_name": "Post-Operative Care", "cert_type": "SKILL",
         "issuing_body": "Hospital Internal", "validity_years": 2,
         "renewal_requires_hours": 20, "renewal_requires_exam": False,
         "is_mandatory": False, "description": "Surgical recovery care"},
        
        {"cert_id": "IV_CERT", "cert_name": "IV Therapy", "cert_type": "SKILL",
         "issuing_body": "Hospital Internal", "validity_years": 2,
         "renewal_requires_hours": 16, "renewal_requires_exam": True,
         "is_mandatory": False, "description": "Intravenous therapy administration"},
        
        {"cert_id": "CHEMO", "cert_name": "Chemotherapy Administration", "cert_type": "SKILL",
         "issuing_body": "Thai Oncology Nursing Society", "validity_years": 2,
         "renewal_requires_hours": 24, "renewal_requires_exam": True,
         "is_mandatory": False, "description": "Safe chemotherapy handling"},
    ]
    
    return pd.DataFrame(certifications)


# =============================================================================
# 4. NURSES
# =============================================================================

def generate_nurses(num_nurses: int, wards_df: pd.DataFrame) -> pd.DataFrame:
    """Generate nurse master data"""
    
    ward_ids = wards_df["ward_id"].tolist()
    
    nurses = []
    for i in range(num_nurses):
        nurse_id = f"N{str(i+1).zfill(4)}"
        
        # Generate realistic attributes
        years_experience = int(np.random.choice(
            range(1, 26), 
            p=np.array([0.15 if y <= 3 else 0.10 if y <= 7 else 0.05 if y <= 15 else 0.02 
                       for y in range(1, 26)]) / sum([0.15 if y <= 3 else 0.10 if y <= 7 else 0.05 if y <= 15 else 0.02 
                       for y in range(1, 26)])
        ))
        
        # Seniority based on experience
        if years_experience >= 15:
            seniority = "SENIOR"
            position = np.random.choice(["HEAD_NURSE", "SENIOR_RN"], p=[0.3, 0.7])
        elif years_experience >= 7:
            seniority = "MID"
            position = "RN"
        else:
            seniority = "JUNIOR"
            position = np.random.choice(["RN", "NEW_GRAD"], p=[0.7, 0.3])
        
        # Employment type
        employment_type = np.random.choice(
            ["FULL_TIME", "PART_TIME", "CONTRACT"],
            p=[0.85, 0.10, 0.05]
        )
        
        # Contracted hours
        if employment_type == "FULL_TIME":
            contracted_hours_week = 40
        elif employment_type == "PART_TIME":
            contracted_hours_week = np.random.choice([20, 24, 32])
        else:
            contracted_hours_week = 40
        
        # Primary ward assignment
        primary_ward = random.choice(ward_ids)
        
        # Secondary wards (can float to)
        ward_type = wards_df[wards_df["ward_id"] == primary_ward]["ward_type"].values[0]
        same_type_wards = wards_df[wards_df["ward_type"] == ward_type]["ward_id"].tolist()
        secondary_wards = [w for w in same_type_wards if w != primary_ward]
        
        # Skill level (1-5)
        skill_level = min(5, max(1, int(years_experience / 5) + np.random.randint(0, 2)))
        
        nurse = {
            "nurse_id": nurse_id,
            "employee_id": f"EMP{str(10000 + i).zfill(6)}",
            "first_name": random.choice(THAI_FIRST_NAMES),
            "last_name": random.choice(THAI_LAST_NAMES),
            "email": f"nurse{i+1}@bumrungrad.com",
            "phone": f"+66{random.randint(800000000, 899999999)}",
            "date_of_birth": (datetime.now() - timedelta(days=365 * random.randint(25, 55))).strftime("%Y-%m-%d"),
            "hire_date": (datetime.now() - timedelta(days=365 * int(years_experience) + random.randint(0, 180))).strftime("%Y-%m-%d"),
            "years_experience": years_experience,
            "seniority": seniority,
            "position": position,
            "employment_type": employment_type,
            "contracted_hours_week": contracted_hours_week,
            "primary_ward_id": primary_ward,
            "secondary_ward_ids": json.dumps(secondary_wards),
            "skill_level": skill_level,
            "can_charge_nurse": years_experience >= 5,
            "can_preceptor": years_experience >= 3,
            "prefers_day_shift": random.random() > 0.3,
            "prefers_night_shift": random.random() > 0.7,
            "max_consecutive_days": 5 if employment_type == "FULL_TIME" else 3,
            "max_night_shifts_week": 3 if not (random.random() > 0.8) else 4,
            "max_weekend_shifts_month": 4,
            "is_active": True,
            "notes": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        
        nurses.append(nurse)
    
    return pd.DataFrame(nurses)


# =============================================================================
# 5. NURSE CERTIFICATIONS (Junction Table)
# =============================================================================

def generate_nurse_certifications(nurses_df: pd.DataFrame, 
                                   certs_df: pd.DataFrame,
                                   wards_df: pd.DataFrame) -> pd.DataFrame:
    """Generate nurse certification holdings with expiry and practice hours"""
    
    nurse_certs = []
    
    for _, nurse in nurses_df.iterrows():
        nurse_id = nurse["nurse_id"]
        primary_ward = nurse["primary_ward_id"]
        years_exp = nurse["years_experience"]
        
        # Everyone has RN and BLS
        mandatory_certs = ["RN", "BLS"]
        
        # Get required certs for primary ward
        ward_info = wards_df[wards_df["ward_id"] == primary_ward].iloc[0]
        required_certs = json.loads(ward_info["requires_certification"])
        
        # Additional certs based on experience
        optional_certs = []
        if years_exp >= 3:
            optional_certs.append("IV_CERT")
        if years_exp >= 5 and random.random() > 0.5:
            optional_certs.append("ACLS")
        if random.random() > 0.8:
            optional_certs.append("CHEMO")
        
        all_certs = list(set(mandatory_certs + required_certs + optional_certs))
        
        for cert_id in all_certs:
            cert_info = certs_df[certs_df["cert_id"] == cert_id].iloc[0]
            
            # Calculate dates
            obtained_date = datetime.now() - timedelta(
                days=random.randint(30, min(int(years_exp) * 365, int(cert_info["validity_years"]) * 365))
            )
            expiry_date = obtained_date + timedelta(days=int(cert_info["validity_years"]) * 365)
            
            # Practice hours this year
            required_hours = int(cert_info["renewal_requires_hours"])
            # Simulate varying progress through the year
            if expiry_date > datetime.now():
                days_until_expiry = (expiry_date - datetime.now()).days
                if days_until_expiry < 90:  # Expiring soon, should have most hours
                    practice_hours_ytd = random.randint(int(required_hours * 0.7), required_hours)
                elif days_until_expiry < 180:
                    practice_hours_ytd = random.randint(int(required_hours * 0.4), int(required_hours * 0.8))
                else:
                    practice_hours_ytd = random.randint(0, int(required_hours * 0.5))
            else:
                practice_hours_ytd = 0  # Expired
            
            nurse_cert = {
                "nurse_cert_id": str(uuid.uuid4()),
                "nurse_id": nurse_id,
                "cert_id": cert_id,
                "obtained_date": obtained_date.strftime("%Y-%m-%d"),
                "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                "is_expired": expiry_date < datetime.now(),
                "practice_hours_ytd": practice_hours_ytd,
                "practice_hours_required": required_hours,
                "hours_remaining": max(0, required_hours - practice_hours_ytd),
                "last_renewal_date": obtained_date.strftime("%Y-%m-%d"),
                "renewal_status": "CURRENT" if expiry_date > datetime.now() else "EXPIRED",
                "verified_by": f"HR_{random.randint(1, 10)}",
                "verification_date": (obtained_date + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
            
            nurse_certs.append(nurse_cert)
    
    return pd.DataFrame(nurse_certs)


# =============================================================================
# 6. SCHEDULING RULES
# =============================================================================

def generate_scheduling_rules() -> pd.DataFrame:
    """Generate scheduling rules/constraints"""
    
    rules = [
        # HARD CONSTRAINTS - Must be satisfied
        {"rule_id": "R001", "rule_name": "Weekly Hours Limit", "rule_type": "HOURS",
         "constraint_type": "HARD", "priority": 1, "scope": "GLOBAL",
         "applies_to_ward": None, "applies_to_position": None,
         "parameters": json.dumps({"max_hours_week": 40, "operator": "<="}),
         "description": "No nurse can work more than 40 hours per week",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R002", "rule_name": "No Overtime", "rule_type": "HOURS",
         "constraint_type": "HARD", "priority": 1, "scope": "GLOBAL",
         "applies_to_ward": None, "applies_to_position": None,
         "parameters": json.dumps({"max_hours_day": 12, "operator": "<="}),
         "description": "Maximum 12 hours per day (no double shifts)",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R003", "rule_name": "Minimum Rest Between Shifts", "rule_type": "REST",
         "constraint_type": "HARD", "priority": 1, "scope": "GLOBAL",
         "applies_to_ward": None, "applies_to_position": None,
         "parameters": json.dumps({"min_rest_hours": 11, "operator": ">="}),
         "description": "Minimum 11 hours rest between shifts",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R004", "rule_name": "Max Consecutive Days", "rule_type": "CONSECUTIVE",
         "constraint_type": "HARD", "priority": 1, "scope": "GLOBAL",
         "applies_to_ward": None, "applies_to_position": None,
         "parameters": json.dumps({"max_consecutive_days": 5, "operator": "<="}),
         "description": "Maximum 5 consecutive working days",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R005", "rule_name": "Minimum Ward Coverage Day", "rule_type": "COVERAGE",
         "constraint_type": "HARD", "priority": 1, "scope": "WARD",
         "applies_to_ward": "ALL", "applies_to_position": None,
         "parameters": json.dumps({"use_ward_minimum": True, "shift": "D"}),
         "description": "Meet minimum staffing for day shift per ward definition",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R006", "rule_name": "Minimum Ward Coverage Evening", "rule_type": "COVERAGE",
         "constraint_type": "HARD", "priority": 1, "scope": "WARD",
         "applies_to_ward": "ALL", "applies_to_position": None,
         "parameters": json.dumps({"use_ward_minimum": True, "shift": "E"}),
         "description": "Meet minimum staffing for evening shift per ward definition",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R007", "rule_name": "Minimum Ward Coverage Night", "rule_type": "COVERAGE",
         "constraint_type": "HARD", "priority": 1, "scope": "WARD",
         "applies_to_ward": "ALL", "applies_to_position": None,
         "parameters": json.dumps({"use_ward_minimum": True, "shift": "N"}),
         "description": "Meet minimum staffing for night shift per ward definition",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R008", "rule_name": "Certification Required", "rule_type": "QUALIFICATION",
         "constraint_type": "HARD", "priority": 1, "scope": "WARD",
         "applies_to_ward": "ALL", "applies_to_position": None,
         "parameters": json.dumps({"check_ward_requirements": True}),
         "description": "Nurses must have required certifications for assigned ward",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R009", "rule_name": "No Night After Evening", "rule_type": "SEQUENCE",
         "constraint_type": "HARD", "priority": 1, "scope": "GLOBAL",
         "applies_to_ward": None, "applies_to_position": None,
         "parameters": json.dumps({"forbidden_sequence": ["E", "N"]}),
         "description": "Cannot work night shift immediately after evening shift",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        # SOFT CONSTRAINTS - Should be optimized
        {"rule_id": "R010", "rule_name": "Certification Practice Hours", "rule_type": "CERTIFICATION",
         "constraint_type": "SOFT", "priority": 2, "scope": "INDIVIDUAL",
         "applies_to_ward": None, "applies_to_position": None,
         "parameters": json.dumps({"ensure_practice_hours": True, "weight": 100}),
         "description": "Prioritize shifts that count toward certification renewal",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R011", "rule_name": "Shift Preference", "rule_type": "PREFERENCE",
         "constraint_type": "SOFT", "priority": 3, "scope": "INDIVIDUAL",
         "applies_to_ward": None, "applies_to_position": None,
         "parameters": json.dumps({"respect_preferences": True, "weight": 50}),
         "description": "Try to honor nurse shift preferences",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R012", "rule_name": "Weekend Fairness", "rule_type": "FAIRNESS",
         "constraint_type": "SOFT", "priority": 2, "scope": "GLOBAL",
         "applies_to_ward": None, "applies_to_position": None,
         "parameters": json.dumps({"balance_weekend_shifts": True, "max_variance": 2, "weight": 80}),
         "description": "Distribute weekend shifts fairly across nurses",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R013", "rule_name": "Night Shift Fairness", "rule_type": "FAIRNESS",
         "constraint_type": "SOFT", "priority": 2, "scope": "GLOBAL",
         "applies_to_ward": None, "applies_to_position": None,
         "parameters": json.dumps({"balance_night_shifts": True, "max_variance": 2, "weight": 80}),
         "description": "Distribute night shifts fairly across nurses",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R014", "rule_name": "Max Night Shifts Per Week", "rule_type": "HOURS",
         "constraint_type": "SOFT", "priority": 2, "scope": "INDIVIDUAL",
         "applies_to_ward": None, "applies_to_position": None,
         "parameters": json.dumps({"use_individual_max": True, "default_max": 3, "weight": 70}),
         "description": "Respect individual max night shifts preference",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R015", "rule_name": "Skill Mix", "rule_type": "COVERAGE",
         "constraint_type": "SOFT", "priority": 2, "scope": "WARD",
         "applies_to_ward": "ALL", "applies_to_position": None,
         "parameters": json.dumps({"min_senior_per_shift": 1, "min_avg_skill_level": 3, "weight": 90}),
         "description": "Each shift should have at least one senior nurse and good skill mix",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R016", "rule_name": "Charge Nurse Coverage", "rule_type": "COVERAGE",
         "constraint_type": "SOFT", "priority": 2, "scope": "WARD",
         "applies_to_ward": "ALL", "applies_to_position": None,
         "parameters": json.dumps({"require_charge_nurse": True, "weight": 95}),
         "description": "Each shift should have a designated charge nurse",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        # ICU-specific rules
        {"rule_id": "R017", "rule_name": "ICU Minimum Ratio", "rule_type": "COVERAGE",
         "constraint_type": "HARD", "priority": 1, "scope": "WARD",
         "applies_to_ward": "W001", "applies_to_position": None,
         "parameters": json.dumps({"nurse_patient_ratio": 0.5, "operator": ">="}),
         "description": "ICU requires 1:2 nurse to patient ratio minimum",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
        
        {"rule_id": "R018", "rule_name": "CCU Minimum Ratio", "rule_type": "COVERAGE",
         "constraint_type": "HARD", "priority": 1, "scope": "WARD",
         "applies_to_ward": "W002", "applies_to_position": None,
         "parameters": json.dumps({"nurse_patient_ratio": 0.5, "operator": ">="}),
         "description": "CCU requires 1:2 nurse to patient ratio minimum",
         "is_active": True, "effective_from": "2024-01-01", "effective_to": None},
    ]
    
    df = pd.DataFrame(rules)
    df["created_at"] = datetime.now()
    df["updated_at"] = datetime.now()
    
    return df


# =============================================================================
# 7. NURSE UNAVAILABILITY / LEAVE REQUESTS
# =============================================================================

def generate_unavailability(nurses_df: pd.DataFrame, 
                            start_date: str,
                            num_weeks: int) -> pd.DataFrame:
    """Generate nurse unavailability and leave requests"""
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = start + timedelta(weeks=num_weeks)
    
    unavailabilities = []
    
    for _, nurse in nurses_df.iterrows():
        nurse_id = nurse["nurse_id"]
        
        # Random leave requests (10% chance per week per nurse)
        current_date = start
        while current_date < end:
            if random.random() < 0.10:  # 10% chance of leave this week
                leave_type = np.random.choice(
                    ["ANNUAL_LEAVE", "SICK_LEAVE", "TRAINING", "PERSONAL", "MATERNITY"],
                    p=[0.50, 0.20, 0.15, 0.10, 0.05]
                )
                
                if leave_type == "MATERNITY":
                    duration = random.randint(30, 90)
                elif leave_type == "TRAINING":
                    duration = random.randint(1, 3)
                elif leave_type == "ANNUAL_LEAVE":
                    duration = random.randint(1, 7)
                else:
                    duration = random.randint(1, 3)
                
                leave_start = current_date + timedelta(days=random.randint(0, 6))
                leave_end = leave_start + timedelta(days=duration)
                
                unavail = {
                    "unavail_id": str(uuid.uuid4()),
                    "nurse_id": nurse_id,
                    "unavail_type": leave_type,
                    "start_date": leave_start.strftime("%Y-%m-%d"),
                    "end_date": min(leave_end, end).strftime("%Y-%m-%d"),
                    "start_time": None,  # All day
                    "end_time": None,
                    "is_recurring": False,
                    "recurrence_pattern": None,
                    "status": np.random.choice(["APPROVED", "PENDING"], p=[0.8, 0.2]),
                    "approved_by": f"HN_{random.randint(1, 5)}" if random.random() > 0.2 else None,
                    "reason": f"{leave_type.replace('_', ' ').title()} request",
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }
                
                unavailabilities.append(unavail)
            
            current_date += timedelta(weeks=1)
        
        # Regular unavailability patterns (e.g., can't work Sundays)
        if random.random() < 0.15:  # 15% have recurring unavailability
            unavail = {
                "unavail_id": str(uuid.uuid4()),
                "nurse_id": nurse_id,
                "unavail_type": "RECURRING",
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
                "start_time": None,
                "end_time": None,
                "is_recurring": True,
                "recurrence_pattern": json.dumps({
                    "day_of_week": random.choice([0, 6]),  # Monday or Sunday
                    "frequency": "WEEKLY"
                }),
                "status": "APPROVED",
                "approved_by": f"HN_{random.randint(1, 5)}",
                "reason": "Regular family commitment",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
            unavailabilities.append(unavail)
    
    return pd.DataFrame(unavailabilities)


# =============================================================================
# 8. SHIFT REQUESTS / PREFERENCES
# =============================================================================

def generate_shift_requests(nurses_df: pd.DataFrame,
                            start_date: str,
                            num_weeks: int) -> pd.DataFrame:
    """Generate specific shift requests from nurses"""
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = start + timedelta(weeks=num_weeks)
    
    requests = []
    
    for _, nurse in nurses_df.iterrows():
        nurse_id = nurse["nurse_id"]
        
        # Each nurse might have 0-3 specific requests
        num_requests = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05])
        
        for _ in range(num_requests):
            request_date = start + timedelta(days=random.randint(0, num_weeks * 7 - 1))
            request_type = np.random.choice(
                ["REQUEST_ON", "REQUEST_OFF", "SWAP_REQUEST", "SHIFT_PREFERENCE"],
                p=[0.3, 0.4, 0.1, 0.2]
            )
            
            req = {
                "request_id": str(uuid.uuid4()),
                "nurse_id": nurse_id,
                "request_type": request_type,
                "request_date": request_date.strftime("%Y-%m-%d"),
                "preferred_shift": random.choice(["D", "E", "N"]) if request_type in ["REQUEST_ON", "SHIFT_PREFERENCE"] else None,
                "swap_with_nurse_id": random.choice(nurses_df["nurse_id"].tolist()) if request_type == "SWAP_REQUEST" else None,
                "priority": np.random.choice(["LOW", "MEDIUM", "HIGH"], p=[0.5, 0.35, 0.15]),
                "reason": random.choice([
                    "Family event", "Medical appointment", "Personal commitment",
                    "Childcare", "Education", "Religious observance", None
                ]),
                "status": np.random.choice(["PENDING", "APPROVED", "DENIED"], p=[0.6, 0.3, 0.1]),
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
            
            requests.append(req)
    
    return pd.DataFrame(requests) if requests else pd.DataFrame()


# =============================================================================
# 9. HISTORICAL ROSTER (for reference/ML)
# =============================================================================

def generate_historical_roster(nurses_df: pd.DataFrame,
                               wards_df: pd.DataFrame,
                               shifts_df: pd.DataFrame,
                               weeks_history: int = 8) -> pd.DataFrame:
    """Generate historical roster data for past weeks"""
    
    end_date = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d") - timedelta(days=1)
    start_date = end_date - timedelta(weeks=weeks_history)
    
    roster_entries = []
    current_date = start_date
    
    working_shifts = shifts_df[shifts_df["shift_type"] == "REGULAR"]["shift_id"].tolist()
    
    while current_date <= end_date:
        for _, nurse in nurses_df.iterrows():
            nurse_id = nurse["nurse_id"]
            ward_id = nurse["primary_ward_id"]
            
            # Simple pattern: ~5 shifts per week
            is_working = random.random() < (5/7)
            
            if is_working:
                # Respect preferences somewhat
                if nurse["prefers_day_shift"] and random.random() < 0.6:
                    shift_id = "D"
                elif nurse["prefers_night_shift"] and random.random() < 0.4:
                    shift_id = "N"
                else:
                    shift_id = random.choice(working_shifts)
                
                shift_info = shifts_df[shifts_df["shift_id"] == shift_id].iloc[0]
                
                entry = {
                    "roster_id": str(uuid.uuid4()),
                    "nurse_id": nurse_id,
                    "ward_id": ward_id,
                    "shift_date": current_date.strftime("%Y-%m-%d"),
                    "shift_id": shift_id,
                    "actual_start_time": shift_info["start_time"],
                    "actual_end_time": shift_info["end_time"],
                    "actual_hours": shift_info["duration_hours"],
                    "is_overtime": False,
                    "overtime_hours": 0,
                    "is_charge_nurse": nurse["can_charge_nurse"] and random.random() < 0.2,
                    "status": "COMPLETED",
                    "absence_type": None,
                    "notes": None,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }
                
                roster_entries.append(entry)
            else:
                # Day off
                entry = {
                    "roster_id": str(uuid.uuid4()),
                    "nurse_id": nurse_id,
                    "ward_id": ward_id,
                    "shift_date": current_date.strftime("%Y-%m-%d"),
                    "shift_id": "OFF",
                    "actual_start_time": None,
                    "actual_end_time": None,
                    "actual_hours": 0,
                    "is_overtime": False,
                    "overtime_hours": 0,
                    "is_charge_nurse": False,
                    "status": "COMPLETED",
                    "absence_type": None,
                    "notes": None,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                }
                
                roster_entries.append(entry)
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(roster_entries)


# =============================================================================
# 10. DEMAND FORECAST
# =============================================================================

def generate_demand_forecast(wards_df: pd.DataFrame,
                             start_date: str,
                             num_weeks: int) -> pd.DataFrame:
    """Generate expected patient census / demand forecast"""
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = start + timedelta(weeks=num_weeks)
    
    forecasts = []
    current_date = start
    
    while current_date < end:
        day_of_week = current_date.weekday()
        
        for _, ward in wards_df.iterrows():
            ward_id = ward["ward_id"]
            bed_count = ward["bed_count"]
            
            # Base occupancy varies by ward type and day
            if ward["ward_type"] == "CRITICAL":
                base_occupancy = 0.85
            elif ward["ward_type"] == "SURGICAL":
                # Lower on weekends (fewer elective surgeries)
                base_occupancy = 0.70 if day_of_week >= 5 else 0.80
            else:
                base_occupancy = 0.75
            
            # Add some noise
            occupancy = min(1.0, max(0.3, base_occupancy + np.random.normal(0, 0.1)))
            expected_patients = int(bed_count * occupancy)
            
            # Acuity varies
            avg_acuity = ward["acuity_level"] + np.random.normal(0, 0.5)
            avg_acuity = min(5, max(1, avg_acuity))
            
            forecast = {
                "forecast_id": str(uuid.uuid4()),
                "ward_id": ward_id,
                "forecast_date": current_date.strftime("%Y-%m-%d"),
                "expected_census": expected_patients,
                "expected_occupancy_pct": round(occupancy * 100, 1),
                "expected_avg_acuity": round(avg_acuity, 2),
                "expected_admissions": random.randint(0, 5),
                "expected_discharges": random.randint(0, 5),
                "confidence_level": np.random.choice(["HIGH", "MEDIUM", "LOW"], p=[0.6, 0.3, 0.1]),
                "forecast_source": "HISTORICAL_MODEL",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
            
            forecasts.append(forecast)
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(forecasts)


# =============================================================================
# 11. SPECIAL EVENTS / CONSTRAINTS
# =============================================================================

def generate_special_events(start_date: str, num_weeks: int) -> pd.DataFrame:
    """Generate special events that affect scheduling"""
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    
    events = [
        {"event_id": "EVT001", "event_name": "JCI Accreditation Visit",
         "event_date": (start + timedelta(days=random.randint(7, 14))).strftime("%Y-%m-%d"),
         "event_type": "INSPECTION", "affected_wards": json.dumps(["ALL"]),
         "staffing_impact": "INCREASE", "impact_percentage": 20,
         "notes": "Ensure full staffing and senior coverage"},
        
        {"event_id": "EVT002", "event_name": "Thai New Year (Songkran)",
         "event_date": (start + timedelta(days=random.randint(0, 28))).strftime("%Y-%m-%d"),
         "event_type": "HOLIDAY", "affected_wards": json.dumps(["ALL"]),
         "staffing_impact": "HIGH_LEAVE_RISK", "impact_percentage": 30,
         "notes": "Many staff request leave during this period"},
        
        {"event_id": "EVT003", "event_name": "Annual Training Day",
         "event_date": (start + timedelta(days=random.randint(14, 21))).strftime("%Y-%m-%d"),
         "event_type": "TRAINING", "affected_wards": json.dumps(["W001", "W002"]),
         "staffing_impact": "REDUCED_AVAILABILITY", "impact_percentage": 15,
         "notes": "ICU and CCU mandatory training"},
    ]
    
    df = pd.DataFrame(events)
    df["created_at"] = datetime.now()
    df["updated_at"] = datetime.now()
    
    return df


# =============================================================================
# MAIN GENERATOR
# =============================================================================

def generate_all_mock_data(output_dir: str = "./mock_data") -> Dict[str, pd.DataFrame]:
    """Generate all mock data tables"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("NURSE ROSTERING MOCK DATA GENERATOR")
    print("=" * 60)
    print(f"Hospital: {CONFIG['hospital_name']}")
    print(f"Nurses: {CONFIG['num_nurses']}")
    print(f"Wards: {CONFIG['num_wards']}")
    print(f"Planning Horizon: {CONFIG['roster_weeks']} weeks starting {CONFIG['start_date']}")
    print("=" * 60)
    
    # Generate in dependency order
    print("\n[1/11] Generating Wards...")
    wards_df = generate_wards()
    
    print("[2/11] Generating Shifts...")
    shifts_df = generate_shifts()
    
    print("[3/11] Generating Certifications...")
    certs_df = generate_certifications()
    
    print("[4/11] Generating Nurses...")
    nurses_df = generate_nurses(CONFIG["num_nurses"], wards_df)
    
    print("[5/11] Generating Nurse Certifications...")
    nurse_certs_df = generate_nurse_certifications(nurses_df, certs_df, wards_df)
    
    print("[6/11] Generating Scheduling Rules...")
    rules_df = generate_scheduling_rules()
    
    print("[7/11] Generating Unavailability/Leave...")
    unavail_df = generate_unavailability(nurses_df, CONFIG["start_date"], CONFIG["roster_weeks"])
    
    print("[8/11] Generating Shift Requests...")
    requests_df = generate_shift_requests(nurses_df, CONFIG["start_date"], CONFIG["roster_weeks"])
    
    print("[9/11] Generating Historical Roster...")
    history_df = generate_historical_roster(nurses_df, wards_df, shifts_df)
    
    print("[10/11] Generating Demand Forecast...")
    demand_df = generate_demand_forecast(wards_df, CONFIG["start_date"], CONFIG["roster_weeks"])
    
    print("[11/11] Generating Special Events...")
    events_df = generate_special_events(CONFIG["start_date"], CONFIG["roster_weeks"])
    
    # Collect all dataframes
    all_data = {
        "dim_wards": wards_df,
        "dim_shifts": shifts_df,
        "dim_certifications": certs_df,
        "dim_nurses": nurses_df,
        "fact_nurse_certifications": nurse_certs_df,
        "dim_scheduling_rules": rules_df,
        "fact_unavailability": unavail_df,
        "fact_shift_requests": requests_df,
        "fact_historical_roster": history_df,
        "fact_demand_forecast": demand_df,
        "dim_special_events": events_df,
    }
    
    # Save to CSV and Parquet
    print("\n" + "=" * 60)
    print("SAVING FILES")
    print("=" * 60)
    
    for name, df in all_data.items():
        csv_path = f"{output_dir}/{name}.csv"
        
        df.to_csv(csv_path, index=False)
        
        print(f"  {name}: {len(df):,} rows -> {csv_path}")
    
    # Generate summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print(f"\n  Total Nurses: {len(nurses_df):,}")
    print(f"    - Full-time: {len(nurses_df[nurses_df['employment_type'] == 'FULL_TIME']):,}")
    print(f"    - Part-time: {len(nurses_df[nurses_df['employment_type'] == 'PART_TIME']):,}")
    print(f"    - Contract: {len(nurses_df[nurses_df['employment_type'] == 'CONTRACT']):,}")
    
    print(f"\n  Nurses by Seniority:")
    for seniority in ["SENIOR", "MID", "JUNIOR"]:
        count = len(nurses_df[nurses_df["seniority"] == seniority])
        print(f"    - {seniority}: {count:,}")
    
    print(f"\n  Nurses by Ward:")
    for _, ward in wards_df.iterrows():
        count = len(nurses_df[nurses_df["primary_ward_id"] == ward["ward_id"]])
        print(f"    - {ward['ward_name']}: {count:,}")
    
    print(f"\n  Total Certifications Held: {len(nurse_certs_df):,}")
    expiring_soon = nurse_certs_df[
        (pd.to_datetime(nurse_certs_df["expiry_date"]) < datetime.now() + timedelta(days=90)) &
        (pd.to_datetime(nurse_certs_df["expiry_date"]) > datetime.now())
    ]
    print(f"    - Expiring in 90 days: {len(expiring_soon):,}")
    
    print(f"\n  Scheduling Rules: {len(rules_df):,}")
    print(f"    - Hard Constraints: {len(rules_df[rules_df['constraint_type'] == 'HARD']):,}")
    print(f"    - Soft Constraints: {len(rules_df[rules_df['constraint_type'] == 'SOFT']):,}")
    
    print(f"\n  Leave/Unavailability Records: {len(unavail_df):,}")
    print(f"  Shift Requests: {len(requests_df):,}")
    print(f"  Historical Roster Entries: {len(history_df):,}")
    print(f"  Demand Forecasts: {len(demand_df):,}")
    
    print("\n" + "=" * 60)
    print("MOCK DATA GENERATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    return all_data


if __name__ == "__main__":
    data = generate_all_mock_data("./mock_data")
