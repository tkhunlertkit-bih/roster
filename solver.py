# solver.py
import argparse
import importlib
import math
import os
from typing import Any, Dict, List

import pandas as pd
from ortools.sat.python import cp_model

from swap_utils import improve_by_day_swaps, improve_by_month_swaps
from validate_roster import validate_roster

ModelT = Any


def load_config(config_name: str) -> Any:
    return importlib.import_module(f"configs.{config_name}")


def create_variables(model, num_nurses: int, days: int, cfg) -> Dict:
    x = {}
    S = len(cfg.SHIFT_NAMES)
    for n in range(num_nurses):
        for d in range(days):
            for s in range(S):
                x[n, d, s] = model.NewBoolVar(f"x_{n}_{d}_{cfg.SHIFT_NAMES[s]}")
    return x


def add_one_shift_per_day(model, x, num_nurses, days, cfg):
    S = len(cfg.SHIFT_NAMES)
    for n in range(num_nurses):
        for d in range(days):
            model.Add(sum(x[n, d, s] for s in range(S)) == 1)


def add_daily_min_staff(model, x, num_nurses, days, cfg):
    S = len(cfg.SHIFT_NAMES)
    work_idx = [i for i in range(S) if cfg.SHIFT_NAMES[i] not in cfg.OFF_SHIFTS]

    for d in range(days):
        model.Add(sum(x[n, d, s] for n in range(num_nurses) for s in work_idx) >= 1)


def add_shift_skill_mix(model, x, nurses_df, days, cfg, min_rn_share=0.35, max_rn_share=0.65):
    """
    Enforce RN share between min_rn_share and max_rn_share
    among (RN+PN) on each day/shift.
    """
    S = len(cfg.SHIFT_NAMES)
    nurse_skills = list(nurses_df["skill"])
    num_nurses = len(nurse_skills)

    for d in range(days):
        for s in range(S):
            rn_terms = []
            pn_terms = []
            for n, skill_n in enumerate(nurse_skills):
                if skill_n == "RN":
                    rn_terms.append(x[n, d, s])
                elif skill_n in ["PN", "SUP"]:
                    pn_terms.append(x[n, d, s])

            if not rn_terms and not pn_terms:
                continue  # no relevant staff for this shift type

            rn_count = model.NewIntVar(0, num_nurses, f"rn_{d}_{s}")
            pn_count = model.NewIntVar(0, num_nurses, f"pn_{d}_{s}")
            total = model.NewIntVar(0, num_nurses, f"tot_{d}_{s}")

            model.Add(rn_count == sum(rn_terms)) if rn_terms else model.Add(rn_count == 0)
            model.Add(pn_count == sum(pn_terms)) if pn_terms else model.Add(pn_count == 0)
            model.Add(total == rn_count + pn_count)

            has_staff = model.NewBoolVar(f"has_staff_{d}_{s}")
            model.Add(total > 0).OnlyEnforceIf(has_staff)
            model.Add(total == 0).OnlyEnforceIf(has_staff.Not())

            # Convert shares to integer inequalities:
            # min_rn_share <= rn/total <= max_rn_share
            # <=> 100*min_rn_share*total <= 100*rn <= 100*max_rn_share*total
            min_pct = int(min_rn_share * 100)
            max_pct = int(max_rn_share * 100)

            model.Add(100 * rn_count >= min_pct * total).OnlyEnforceIf(has_staff)
            model.Add(100 * rn_count <= max_pct * total).OnlyEnforceIf(has_staff)


def add_staff_to_patient_ratios_by_block(model, x, nurses_df, days, cfg, beds_df):
    """
    Hard safety constraints: enforce RN/PN minimum counts per block (B1..B4)
    based on beds_per_day and staff-to-patient ratios.
    Uses SHIFT_BLOCKS so that any shift that covers a block contributes there.
    """
    shift_index = {name: i for i, name in enumerate(cfg.SHIFT_NAMES)}
    nurse_skills = list(nurses_df["skill"])

    # Ratios from guidelines (per block, approximate day vs night)
    # B1, B2, B3 treated as "day"; B4 as "night"
    def day_rn_req(beds: int) -> int:
        return (beds + 5 - 1) // 5  # RN day 1:5

    def day_pn_req(beds: int) -> int:
        return (beds + 6 - 1) // 6  # PN day 1:6

    def night_rn_req(beds: int) -> int:
        return (beds + 6 - 1) // 6  # RN night 1:6

    def night_pn_req(beds: int) -> int:
        return (beds + 7 - 1) // 7  # PN night 1:7 (safe side of 7–10)

    for d in range(days):
        beds = int(beds_df.loc[beds_df["day"] == d + 1, "beds"].iloc[0])

        for block in cfg.TIME_BLOCKS:
            rn_terms = []
            pn_terms = []

            # Collect all x[n,d,s] whose shift covers this block
            for n, skill_n in enumerate(nurse_skills):
                for s_name in cfg.SHIFT_NAMES:
                    if block in cfg.SHIFT_BLOCKS[s_name]:
                        s = shift_index[s_name]
                        if skill_n == "RN":
                            rn_terms.append(x[n, d, s])
                        elif skill_n in ["PN", "SUP"]:
                            pn_terms.append(x[n, d, s])

            if not rn_terms and not pn_terms:
                continue

            # Decide if this block is 'day' or 'night' for ratio purposes
            if block in ["B1", "B2", "B3"]:
                rn_req = day_rn_req(beds)
                pn_req = day_pn_req(beds)
            else:  # B4
                rn_req = night_rn_req(beds)
                pn_req = night_pn_req(beds)

            if rn_terms:
                model.Add(sum(rn_terms) >= rn_req)
            if pn_terms:
                model.Add(sum(pn_terms) >= pn_req)


def add_hours_constraints_and_excess(model, x, nurses_df, days, cfg):
    """
    - Enforce monthly min/max hours per nurse.
    - Create h_n (total hours) and e_n (excess over 184) for later use.
    Returns:
        total_hours_vars: dict[n_idx] -> IntVar
        excess_hours_vars: dict[n_idx] -> IntVar
    """
    S = len(cfg.SHIFT_NAMES)
    total_hours_vars = {}
    excess_hours_vars = {}

    for n_idx, row in nurses_df.iterrows():
        min_h = int(row["monthly_min_hours"])
        max_h = int(row["monthly_max_hours"])
        threshold = 184  # business rule for “expected” hours

        total_hours = model.NewIntVar(0, max_h * 2, f"hours_{n_idx}")
        model.Add(
            total_hours
            == sum(cfg.SHIFT_HOURS[cfg.SHIFT_NAMES[s]] * x[n_idx, d, s] for d in range(days) for s in range(S))
        )
        model.Add(total_hours >= min_h)
        model.Add(total_hours <= max_h)
        total_hours_vars[n_idx] = total_hours

        excess = model.NewIntVar(0, max_h * 2, f"excess_{n_idx}")
        model.Add(excess >= total_hours - threshold)
        model.Add(excess >= 0)
        excess_hours_vars[n_idx] = excess

    return total_hours_vars, excess_hours_vars


def add_leave_constraints(model, x, nurses_df, prefs_df, days, cfg, pub_days_per_nurse: int):
    shift_index = {name: i for i, name in enumerate(cfg.SHIFT_NAMES)}
    N = len(nurses_df)

    bd_idx = shift_index["BD"]
    v_idx = shift_index["V"]
    pub_idx = shift_index["PUB"]

    pub_totals = []
    for n in range(N):
        model.Add(sum(x[n, d, bd_idx] for d in range(days)) <= cfg.MAX_BD_PER_MONTH)

        for d in range(days):
            pref = prefs_df.loc[n, str(d + 1)]
            if pref != "V":
                model.Add(x[n, d, v_idx] == 0)

        pub_total = model.NewIntVar(0, days, f"pub_total_{n}")
        model.Add(pub_total == sum(x[n, d, pub_idx] for d in range(days)))
        model.Add(pub_total == pub_days_per_nurse)
        pub_totals.append(pub_total)


def add_forbidden_transitions(model, x, num_nurses, days, cfg):
    shift_index = {name: i for i, name in enumerate(cfg.SHIFT_NAMES)}
    for n in range(num_nurses):
        for d in range(days - 1):
            for a, bad_next in cfg.FORBIDDEN_NEXT.items():
                sa = shift_index[a]
                for b in bad_next:
                    sb = shift_index[b]
                    model.Add(x[n, d, sa] + x[n, d + 1, sb] <= 1)


def add_days_off_rules(model, x, num_nurses, days, cfg):
    S = len(cfg.SHIFT_NAMES)
    shift_index = {name: i for i, name in enumerate(cfg.SHIFT_NAMES)}
    off_idx = [shift_index[s] for s in cfg.OFF_SHIFTS]
    work_idx = [i for i in range(S) if cfg.SHIFT_NAMES[i] not in cfg.OFF_SHIFTS]

    for n in range(num_nurses):
        for d in range(days - cfg.MAX_CONSEC_WORK_DAYS):
            flags = []
            for k in range(cfg.MAX_CONSEC_WORK_DAYS):
                flag = model.NewBoolVar(f"work_{n}_{d+k}")
                model.Add(sum(x[n, d + k, s] for s in work_idx) == 1).OnlyEnforceIf(flag)
                model.Add(sum(x[n, d + k, s] for s in work_idx) != 1).OnlyEnforceIf(flag.Not())
                flags.append(flag)
            all_work = model.NewBoolVar(f"work_streak_{n}_{d}")
            model.Add(sum(flags) == cfg.MAX_CONSEC_WORK_DAYS).OnlyEnforceIf(all_work)
            model.Add(sum(flags) != cfg.MAX_CONSEC_WORK_DAYS).OnlyEnforceIf(all_work.Not())
            off_next = sum(x[n, d + cfg.MAX_CONSEC_WORK_DAYS, s] for s in off_idx)
            model.Add(off_next == 1).OnlyEnforceIf(all_work)


def build_preference_terms(model, x, nurses_df, prefs_df, days, cfg):
    shift_index = {name: i for i, name in enumerate(cfg.SHIFT_NAMES)}
    N = len(nurses_df)

    penalties = []
    dayoff_penalties = []
    long_penalty_terms = []
    long_pref_matches = []
    requesters = []

    for n in range(N):
        terms = []
        dayoff_terms = []
        has_pref = False

        for d in range(days):
            pref = prefs_df.loc[n, str(d + 1)]
            if pd.isna(pref) or pref == "":
                continue
            has_pref = True
            pref = str(pref)
            if pref == "R":
                pref = "X"

            if pref == "X":
                # requested day off
                is_off = model.NewBoolVar(f"is_off_{n}_{d}")
                off_lits = [x[n, d, shift_index[s]] for s in cfg.OFF_SHIFTS if s in shift_index]
                if off_lits:
                    model.Add(sum(off_lits) == 1).OnlyEnforceIf(is_off)
                    model.Add(sum(off_lits) != 1).OnlyEnforceIf(is_off.Not())
                else:
                    model.Add(is_off == 0)
                dayoff_violation = model.NewBoolVar(f"dayoff_violation_{n}_{d}")
                model.Add(is_off == 0).OnlyEnforceIf(dayoff_violation)
                model.Add(is_off == 1).OnlyEnforceIf(dayoff_violation.Not())
                dayoff_terms.append(dayoff_violation)
            else:
                # normal shift preference
                if pref in shift_index:
                    s_pref = shift_index[pref]
                    mismatch = model.NewBoolVar(f"mismatch_{n}_{d}")
                    model.Add(x[n, d, s_pref] == 0).OnlyEnforceIf(mismatch)
                    model.Add(x[n, d, s_pref] == 1).OnlyEnforceIf(mismatch.Not())
                    terms.append(mismatch)

                    match = model.NewBoolVar(f"match_{n}_{d}")
                    model.Add(x[n, d, s_pref] == 1).OnlyEnforceIf(match)
                    model.Add(x[n, d, s_pref] == 0).OnlyEnforceIf(match.Not())
                    if pref in cfg.LONG_SHIFTS:
                        long_pref_matches.append(match)

        total_penalty = model.NewIntVar(0, days * 2, f"penalty_{n}")
        model.Add(total_penalty == sum(terms)) if terms else model.Add(total_penalty == 0)
        penalties.append(total_penalty)

        dayoff_penalty = model.NewIntVar(0, days, f"dayoff_penalty_{n}")
        model.Add(dayoff_penalty == sum(dayoff_terms)) if dayoff_terms else model.Add(dayoff_penalty == 0)
        dayoff_penalties.append(dayoff_penalty)

        if has_pref:
            requesters.append(n)

    # long_shift generic penalty (optional; you set W_LONG=0 so this is unused)
    shift_index = {name: i for i, name in enumerate(cfg.SHIFT_NAMES)}
    for n in range(N):
        for d in range(days):
            is_long = model.NewBoolVar(f"is_long_{n}_{d}")
            long_lits = [x[n, d, shift_index[s_name]] for s_name in cfg.LONG_SHIFTS if s_name in shift_index]
            if long_lits:
                long_sum = model.NewIntVar(0, len(long_lits), f"long_sum_{n}_{d}")
                model.Add(long_sum == sum(long_lits))
                model.Add(long_sum >= 1).OnlyEnforceIf(is_long)
                model.Add(long_sum == 0).OnlyEnforceIf(is_long.Not())
                long_penalty_terms.append(is_long)

    return penalties, dayoff_penalties, long_penalty_terms, long_pref_matches, requesters


def add_preference_fair_objective(
    model, x, nurses_df, prefs_df, days, cfg, excess_hours_vars, total_hours_all, beds_sum, fte_uos_threshold
):
    penalties, dayoff_penalties, long_penalty_terms, long_pref_matches, requesters = build_preference_terms(
        model, x, nurses_df, prefs_df, days, cfg
    )

    # fairness over combined denies (if you still want it; W_FAIR can be 0)
    N = len(nurses_df)
    denies = []
    for n in range(N):
        combined = model.NewIntVar(0, days * 2, f"combined_deny_{n}")
        model.Add(combined == penalties[n] + dayoff_penalties[n])
        denies.append(combined)

    R = len(requesters)
    if R > 0:
        avg_penalty = model.NewIntVar(0, days * 2, "avg_penalty")
        model.Add(avg_penalty * R == sum(denies[n] for n in requesters))

        deviations = []
        for n in requesters:
            p = denies[n]
            dev = model.NewIntVar(0, days * 2, f"dev_{n}")
            model.Add(dev >= p - avg_penalty)
            model.Add(dev >= avg_penalty - p)
            deviations.append(dev)
    else:
        deviations = []

    # FTE/UOS soft penalty (unchanged)
    target_max_hours = int(fte_uos_threshold * beds_sum * 8)
    over_fte = model.NewIntVar(0, 2 * target_max_hours, "over_fte")
    model.Add(over_fte >= total_hours_all - target_max_hours)
    model.Add(over_fte >= 0)

    W_HOURS = cfg.EXCEEDING_HOURS_WEIGHT
    W_DAYOFF = cfg.DENIED_DAYS_OFF_WEIGHT
    W_PREF = cfg.PREFERENCE_WEIGHT
    W_FAIR = cfg.FAIRNESS_WEIGHT
    W_FTE = cfg.OVER_FTE_WEIGHT
    W_LONG = cfg.LONG_SHIFT_PENALTY
    W_LONG_PREF = cfg.PREF_REWARD

    model.Minimize(
        W_HOURS * sum(excess_hours_vars.values())
        + W_DAYOFF * sum(dayoff_penalties)
        + W_PREF * sum(penalties)
        + W_FAIR * sum(deviations)
        + W_LONG * sum(long_penalty_terms)
        - W_LONG_PREF * sum(long_pref_matches)
        + W_FTE * over_fte
    )

    # return for lexicographic use
    return penalties, dayoff_penalties


def add_long_off_vacation_rule(model, x, num_nurses, days, cfg):
    shift_index = {name: i for i, name in enumerate(cfg.SHIFT_NAMES)}
    off_idx = [shift_index[s] for s in cfg.OFF_SHIFTS]
    v_idx = shift_index["V"]

    for n in range(num_nurses):
        # For each possible start of a 4+ off-day run
        for d in range(days - 3):
            # indicator: days d..d+3 are all off
            flags = []
            for k in range(4):
                is_off = model.NewBoolVar(f"is_off_{n}_{d+k}")
                model.Add(sum(x[n, d + k, s] for s in off_idx) == 1).OnlyEnforceIf(is_off)
                model.Add(sum(x[n, d + k, s] for s in off_idx) != 1).OnlyEnforceIf(is_off.Not())
                flags.append(is_off)

            four_off = model.NewBoolVar(f"four_off_{n}_{d}")
            model.Add(sum(flags) == 4).OnlyEnforceIf(four_off)
            model.Add(sum(flags) != 4).OnlyEnforceIf(four_off.Not())

            # If there is at least a 4-day off run starting here,
            # then from day d+3 onward, as long as the streak continues,
            # all those days must be V.
            # For simplicity: enforce V at day d+3 when four_off is true.
            model.Add(x[n, d + 3, v_idx] == 1).OnlyEnforceIf(four_off)


def compute_request_stats(roster_df, prefs_df, days):
    """
    Returns:
      req_stats: list of dicts per nurse with counts
      denied_rows: list of dicts for each denied request
    """
    req_stats = []
    denied_rows = []

    # Ensure prefs_df indexed by nurse_id
    if "nurse_id" in prefs_df.columns:
        prefs = prefs_df.set_index("nurse_id")
    else:
        # fallback: assume same order
        prefs = prefs_df.copy()
        prefs["nurse_id"] = roster_df["nurse_id"]
        prefs = prefs.set_index("nurse_id")

    for _, r in roster_df.iterrows():
        nid = r["nurse_id"]
        total_req = 0
        accepted = 0
        denied = 0

        for d in range(1, days + 1):
            col = str(d)
            pref = prefs.loc[nid, col]
            assg = r[col]

            # normalize NaN / empty
            if pd.isna(pref):
                pref = ""
            # treat R or PUB as X, day off.
            elif pref in ("R"):
                pref = "X"
            pref = str(pref)

            if assg == "PUB":
                assg = "X"

            # day-off requests (X/R) are tracked separately in the objective,
            # skip here to avoid double-counting
            if pref == "":
                continue

            total_req += 1
            if assg == pref:
                accepted += 1
            else:
                denied += 1
                denied_rows.append(
                    {
                        "nurse_id": nid,
                        "day": d,
                        "requested": pref,
                        "assigned": assg,
                    }
                )

        req_stats.append(
            {
                "nurse_id": nid,
                "total_requests": total_req,
                "accepted": accepted,
                "denied": denied,
            }
        )

    return pd.DataFrame(req_stats), pd.DataFrame(denied_rows)


def solve_stage1_min_x(cfg, input_dir, month_days, pub_days_per_nurse, max_time_sec):
    nurses_path = os.path.join(input_dir, "nurses.csv")
    prefs_path = os.path.join(input_dir, "preferences.csv")
    beds_path = os.path.join(input_dir, "beds_per_day.csv")

    nurses_df = pd.read_csv(nurses_path)
    prefs_df = pd.read_csv(prefs_path)
    beds_df = pd.read_csv(beds_path)

    num_nurses = len(nurses_df)
    days = month_days

    model: ModelT = cp_model.CpModel()
    x = create_variables(model, num_nurses, days, cfg)

    add_one_shift_per_day(model, x, num_nurses, days, cfg)
    add_daily_min_staff(model, x, num_nurses, days, cfg)
    add_shift_skill_mix(model, x, nurses_df, days, cfg)
    add_staff_to_patient_ratios_by_block(model, x, nurses_df, days, cfg, beds_df)
    total_hours_vars, excess_hours_vars = add_hours_constraints_and_excess(model, x, nurses_df, days, cfg)
    add_leave_constraints(model, x, nurses_df, prefs_df, days, cfg, pub_days_per_nurse)
    add_forbidden_transitions(model, x, num_nurses, days, cfg)
    add_days_off_rules(model, x, num_nurses, days, cfg)

    # Build preference terms but define our own objective
    _, dayoff_penalties, _, _, _ = build_preference_terms(model, x, nurses_df, prefs_df, days, cfg)

    # Stage 1 objective: primarily minimize X-denials, also keep excess hours small
    W_HOURS = cfg.EXCEEDING_HOURS_WEIGHT
    W_DAYOFF = cfg.DENIED_DAYS_OFF_WEIGHT

    model.Minimize(W_DAYOFF * sum(dayoff_penalties) + W_HOURS * sum(excess_hours_vars.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_sec

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution in stage 1")

    min_x_denials = int(sum(solver.Value(v) for v in dayoff_penalties))
    # extract roster
    data = {"nurse_id": list(nurses_df["nurse_id"])}
    for d in range(days):
        col = []
        for n in range(num_nurses):
            for s_idx, s_name in enumerate(cfg.SHIFT_NAMES):
                if solver.BooleanValue(x[n, d, s_idx]):
                    col.append(s_name)
                    break
        data[str(d + 1)] = col
    roster_df = pd.DataFrame(data)
    roster_df["total_hours"] = [int(solver.Value(total_hours_vars[n])) for n in range(num_nurses)]
    roster_df["exceed_hours"] = [int(solver.Value(excess_hours_vars[n])) for n in range(num_nurses)]

    return model, x, solver, nurses_df, prefs_df, beds_df, total_hours_vars, excess_hours_vars, min_x_denials, roster_df


def solve_stage2_with_x_bound(
    cfg, input_dir, month_days, pub_days_per_nurse, fte_uos_threshold, max_time_sec, min_x_denials
):
    nurses_path = os.path.join(input_dir, "nurses.csv")
    prefs_path = os.path.join(input_dir, "preferences.csv")
    beds_path = os.path.join(input_dir, "beds_per_day.csv")

    nurses_df = pd.read_csv(nurses_path)
    prefs_df = pd.read_csv(prefs_path)
    beds_df = pd.read_csv(beds_path)

    num_nurses = len(nurses_df)
    days = month_days

    model: ModelT = cp_model.CpModel()
    x = create_variables(model, num_nurses, days, cfg)

    add_one_shift_per_day(model, x, num_nurses, days, cfg)
    add_daily_min_staff(model, x, num_nurses, days, cfg)
    add_shift_skill_mix(model, x, nurses_df, days, cfg)
    add_staff_to_patient_ratios_by_block(model, x, nurses_df, days, cfg, beds_df)
    total_hours_vars, excess_hours_vars = add_hours_constraints_and_excess(model, x, nurses_df, days, cfg)
    add_leave_constraints(model, x, nurses_df, prefs_df, days, cfg, pub_days_per_nurse)
    add_forbidden_transitions(model, x, num_nurses, days, cfg)
    add_days_off_rules(model, x, num_nurses, days, cfg)

    total_hours_all = sum(total_hours_vars.values())
    beds_sum = int(beds_df["beds"].sum())

    _, dayoff_penalties, _, _, _ = build_preference_terms(model, x, nurses_df, prefs_df, days, cfg)

    # Hard bound: do not allow more X denials than min_x_denials
    model.Add(sum(dayoff_penalties) <= min_x_denials)

    # Full objective as before (or with W_FTE=1, W_LONG=0 etc.)
    add_preference_fair_objective(
        model, x, nurses_df, prefs_df, days, cfg, excess_hours_vars, total_hours_all, beds_sum, fte_uos_threshold
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_sec

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution in stage 2")

    # extract roster
    data = {"nurse_id": list(nurses_df["nurse_id"])}
    for d in range(days):
        col = []
        for n in range(num_nurses):
            for s_idx, s_name in enumerate(cfg.SHIFT_NAMES):
                if solver.BooleanValue(x[n, d, s_idx]):
                    col.append(s_name)
                    break
        data[str(d + 1)] = col
    roster_df = pd.DataFrame(data)
    roster_df["total_hours"] = [int(solver.Value(total_hours_vars[n])) for n in range(num_nurses)]
    roster_df["exceed_hours"] = [int(solver.Value(excess_hours_vars[n])) for n in range(num_nurses)]

    return roster_df


def solve_month_with_solution(
    cfg,
    input_dir: str,
    month_days: int,
    pub_days_per_nurse: int,
    fte_uos_threshold: float,
    max_time_sec: float = 60.0,
):
    nurses_path = os.path.join(input_dir, "nurses.csv")
    prefs_path = os.path.join(input_dir, "preferences.csv")
    beds_path = os.path.join(input_dir, "beds_per_day.csv")

    nurses_df = pd.read_csv(nurses_path)
    prefs_df = pd.read_csv(prefs_path)
    beds_df = pd.read_csv(beds_path)
    # coverage_blocks = load_coverage_blocks(coverage_blocks_path)

    num_nurses = len(nurses_df)
    days = month_days

    model: ModelT = cp_model.CpModel()
    x = create_variables(model, num_nurses, days, cfg)
    add_one_shift_per_day(model, x, num_nurses, days, cfg)
    add_daily_min_staff(model, x, num_nurses, days, cfg)
    # add_long_off_vacation_rule(model, x, num_nurses, days, cfg)
    add_shift_skill_mix(model, x, nurses_df, days, cfg)
    add_staff_to_patient_ratios_by_block(model, x, nurses_df, days, cfg, beds_df)
    total_hours_vars, excess_hours_vars = add_hours_constraints_and_excess(model, x, nurses_df, days, cfg)
    add_leave_constraints(model, x, nurses_df, prefs_df, days, cfg, pub_days_per_nurse)
    add_forbidden_transitions(model, x, num_nurses, days, cfg)
    add_days_off_rules(model, x, num_nurses, days, cfg)

    total_hours_all = sum(total_hours_vars.values())
    beds_sum = int(beds_df["beds"].sum())
    # max_hours = int(fte_uos_threshold * beds_sum * 8)
    # model.Add(total_hours_all <= max_hours)
    add_preference_fair_objective(
        model, x, nurses_df, prefs_df, days, cfg, excess_hours_vars, total_hours_all, beds_sum, fte_uos_threshold
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_sec
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(status)
        raise RuntimeError("No feasible solution found")

    data: Dict[str, List[str]] = {"nurse_id": list(nurses_df["nurse_id"])}
    for d in range(days):
        col: List[str] = []
        for n in range(num_nurses):
            for s_idx, s_name in enumerate(cfg.SHIFT_NAMES):
                if solver.BooleanValue(x[n, d, s_idx]):
                    col.append(s_name)
                    break
        data[str(d + 1)] = col
    roster_df = pd.DataFrame(data)

    # hours & excess (for output)
    total_hours = []
    excess_hours = []
    for n in range(num_nurses):
        total_hours.append(int(solver.Value(total_hours_vars[n])))
        excess_hours.append(int(solver.Value(excess_hours_vars[n])))

    roster_df["total_hours"] = total_hours
    roster_df["exceed_hours"] = excess_hours
    return roster_df, (model, x, solver, nurses_df, prefs_df, beds_df, total_hours_vars)


def improve_with_second_pass(
    cfg,
    input_dir,
    month_days,
    pub_days_per_nurse,
    fte_uos_threshold,
    max_time_sec=60.0,
) -> pd.DataFrame:
    # --- first pass: current behaviour ---
    try:
        roster_df, first_solution = solve_month_with_solution(
            cfg, input_dir, month_days, pub_days_per_nurse, fte_uos_threshold, max_time_sec
        )
    except RuntimeError:
        # if the solution is infeasible, just return empty roster.
        return pd.DataFrame()

    # first_solution should include: x, model, total_hours_vars, etc.
    _, x1, solver_f, nurses_df, prefs_df, beds_df, total_hours_vars1 = first_solution

    # store first-pass work/off pattern and total_hours as constants
    num_nurses = len(nurses_df)
    days = month_days
    S = len(cfg.SHIFT_NAMES)
    work_shifts = [s for s, name in enumerate(cfg.SHIFT_NAMES) if name not in cfg.OFF_SHIFTS]

    work_off = {}
    hours_target = {}
    for n in range(num_nurses):
        hours_target[n] = int(solver_f.Value(total_hours_vars1[n]))
        for d in range(days):
            worked = any(solver_f.Value(x1[n, d, s]) for s in work_shifts)
            work_off[n, d] = int(worked)

    # --- second pass: rebuild model with locks ---
    model2: ModelT = cp_model.CpModel()
    x2 = create_variables(model2, num_nurses, days, cfg)

    add_one_shift_per_day(model2, x2, num_nurses, days, cfg)
    add_daily_min_staff(model2, x2, num_nurses, days, cfg)
    add_shift_skill_mix(model2, x2, nurses_df, days, cfg)
    add_staff_to_patient_ratios_by_block(model2, x2, nurses_df, days, cfg, beds_df)

    # hours constraints with fixed total hours
    total_hours_vars2, excess_hours_vars2 = add_hours_constraints_and_excess(model2, x2, nurses_df, days, cfg)
    for n in range(num_nurses):
        model2.Add(total_hours_vars2[n] == hours_target[n])

    add_leave_constraints(model2, x2, nurses_df, prefs_df, days, cfg, pub_days_per_nurse)
    add_forbidden_transitions(model2, x2, num_nurses, days, cfg)
    add_days_off_rules(model2, x2, num_nurses, days, cfg)

    # lock work/off pattern from first pass
    work_idx = [i for i in range(S) if cfg.SHIFT_NAMES[i] not in cfg.OFF_SHIFTS]

    for n in range(num_nurses):
        for d in range(days):
            if work_off[n, d] == 0:
                model2.Add(sum(x2[n, d, s] for s in work_idx) == 0)
            else:
                model2.Add(sum(x2[n, d, s] for s in work_idx) == 1)

    # objective now: mostly preferences
    total_hours_all2 = sum(total_hours_vars2.values())
    beds_sum = int(beds_df["beds"].sum())
    add_preference_fair_objective(
        model2, x2, nurses_df, prefs_df, days, cfg, excess_hours_vars2, total_hours_all2, beds_sum, fte_uos_threshold
    )

    solver2 = cp_model.CpSolver()
    solver2.parameters.max_time_in_seconds = max_time_sec
    status2 = solver2.Solve(model2)
    if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # fallback to roster_df from first pass
        return roster_df

    # extract improved roster
    data = {"nurse_id": list(nurses_df["nurse_id"])}
    for d in range(days):
        col = []
        for n in range(num_nurses):
            for s_idx, s_name in enumerate(cfg.SHIFT_NAMES):
                if solver2.BooleanValue(x2[n, d, s_idx]):
                    col.append(s_name)
                    break
        data[str(d + 1)] = col
    improved_roster_df = pd.DataFrame(data)
    improved_roster_df["total_hours"] = [int(solver2.Value(total_hours_vars2[n])) for n in range(num_nurses)]
    improved_roster_df["exceed_hours"] = [int(solver2.Value(excess_hours_vars2[n])) for n in range(num_nurses)]

    return improved_roster_df


def lexicographic_solve(cfg, input_dir, month_days, pub_days_per_nurse, fte_uos_threshold, max_time_sec):
    _, _, _, _, _, _, _, _, min_x_denial, roster_df = solve_stage1_min_x(
        cfg, input_dir, month_days, pub_days_per_nurse, max_time_sec
    )
    print(f"min_x_denials: {min_x_denial}")
    try:
        roster_df = solve_stage2_with_x_bound(
            cfg, input_dir, month_days, pub_days_per_nurse, fte_uos_threshold, max_time_sec, min_x_denial + 5
        )
    except RuntimeError:
        print("unable to solve 2, reverting back to solve 1 solution")
    return roster_df


def score_roster(roster_df, prefs_df, days, cfg):
    if roster_df.empty:
        return math.inf

    _, denied_rows = compute_request_stats(roster_df, prefs_df, days)

    # categories:
    #  - X -> work (very bad)
    #  - work -> X (bad)
    #  - shift -> other shift (base bad)
    #  - M <-> M4/ME, N <-> N4 (0.5)
    VERY_BAD = 3.0
    BAD = 2.0
    BASE = 1.0
    HALF = 0.5

    score = 0.0

    for _, row in denied_rows.iterrows():
        req = row["requested"]
        assg = row["assigned"]

        is_off_req = req == "X"
        is_off_assg = assg in cfg.OFF_SHIFTS

        # X -> work
        if is_off_req and not is_off_assg:
            score += VERY_BAD
            continue

        # work -> X
        if not is_off_req and is_off_assg:
            score += BAD
            continue

        # shift -> other shift
        # group mornings and nights for 0.5 weight
        morning_group = {"M", "M4", "ME"}
        night_group = {"N", "N4"}

        if req in morning_group and assg in morning_group:
            score += HALF
        elif req in night_group and assg in night_group:
            score += HALF
        else:
            score += BASE

    return score


def multi_seed_best_roster(
    cfg,
    nurses_df,
    beds_df,
    input_dir,
    month_days,
    pub_days_per_nurse,
    fte_uos_threshold,
    days,
    num_seeds=5,
    max_time_sec=60,
):
    prefs_path = os.path.join(input_dir, "preferences.csv")
    prefs_df = pd.read_csv(prefs_path)

    best_score = None
    best_roster = pd.DataFrame()

    for seed in range(num_seeds):
        print(f"iteration: {seed}")
        # set seed before each solve
        # cp_solver = cp_model.CpSolver()
        # cp_solver.random_seed = seed  # you can instead set via global param if you refactor

        # For simplicity, set seed inside each stage function
        roster_df = lexicographic_solve(cfg, input_dir, month_days, pub_days_per_nurse, fte_uos_threshold, max_time_sec)
        ok1 = validate_roster(roster_df, nurses_df, beds_df, cfg, days)
        if not ok1:
            print("shit, failed lexicographic before swap attempts")
            exit()
        roster_df = improve_by_month_swaps(roster_df, nurses_df, prefs_df, cfg, days)
        roster_df = improve_by_day_swaps(roster_df, nurses_df, prefs_df, cfg, days)
        ok = validate_roster(roster_df, nurses_df, beds_df, cfg, days)

        s = score_roster(roster_df, prefs_df, days, cfg)
        print(f"score lexico: {s}")
        if (best_score is None or s < best_score) and ok:
            print("better model saved")
            best_score = s
            best_roster = roster_df

        roster2_df = improve_with_second_pass(
            cfg, input_dir, month_days, pub_days_per_nurse, fte_uos_threshold, max_time_sec
        )
        if not roster2_df.empty:
            ok2 = validate_roster(roster2_df, nurses_df, beds_df, cfg, days)
            if not ok2:
                print("shit, failed improve_with_second_pass before swap attempts")
                exit()
            roster2_df = improve_by_month_swaps(roster2_df, nurses_df, prefs_df, cfg, days)
            roster2_df = improve_by_day_swaps(roster2_df, nurses_df, prefs_df, cfg, days)
            ok = validate_roster(roster2_df, nurses_df, beds_df, cfg, days)

            s2 = score_roster(roster2_df, prefs_df, days, cfg)
            print(f"score 2-pass: {s2}")
            if (best_score is None or s2 < best_score) and ok:
                print("better model saved")
                best_score = s2
                best_roster = roster2_df
        print()

    return best_roster, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ipd_nurse")
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Directory with nurses.csv, preferences.csv",
    )
    parser.add_argument(
        "--pub_days_per_nurse",
        type=int,
        required=True,
        help="Number of public holiday of this month",
    )
    parser.add_argument(
        "--fte_uos_threshold",
        type=float,
        required=True,
        help="FTE/UOS cap of this month",
    )
    parser.add_argument("--days", type=int, default=31)
    parser.add_argument("--max_time", type=float, default=60.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_dir = args.input_dir or os.path.join("mock_data", args.config)

    nurses_path = os.path.join(input_dir, "nurses.csv")
    nurses_df = pd.read_csv(nurses_path)
    beds_path = os.path.join(input_dir, "beds_per_day.csv")
    beds_df = pd.read_csv(beds_path)

    roster_df, score = multi_seed_best_roster(
        cfg,
        nurses_df,
        beds_df,
        input_dir=input_dir,
        month_days=args.days,
        pub_days_per_nurse=args.pub_days_per_nurse,
        fte_uos_threshold=args.fte_uos_threshold,
        days=args.days,
        max_time_sec=args.max_time,
        num_seeds=10,
    )

    ok = validate_roster(roster_df, nurses_df, beds_df, cfg, args.days)
    if not ok:
        exit()

    # Merge nurse info
    merged = roster_df.merge(nurses_df, on="nurse_id")

    # Build display columns
    threshold = 184
    name_phone = merged["name"] + "\n" + merged["phone_number"].astype(str)
    flag_over_240 = (merged["exceed_hours"] > 240).astype(int)

    # Column order
    day_cols = [str(d) for d in range(1, args.days + 1)]
    for d in day_cols:
        merged[d] = merged[d].replace("R", "X")

    final_df = pd.DataFrame(
        {
            "name_phone": name_phone,
            "level": merged["level"],
            "role": merged["skill"],  # or merged["skill"] if you prefer
            "nurse_id": merged["nurse_id"],
            "threshold_hours": threshold,
            "total_hours": merged["total_hours"],
            "exceed_hours": merged["exceed_hours"],
            "flag_over_240": flag_over_240,
        }
    )

    for d in day_cols:
        final_df[d] = merged[d]

    out_path = os.path.join(input_dir, "roster_output.xlsx")
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, index=False, sheet_name="Roster")

        # Block headcount sheet
        block_rows = []
        for day in range(1, args.days + 1):
            col = str(day)
            for block in cfg.TIME_BLOCKS:
                total = 0
                rn = 0
                pn = 0
                for _, row in merged.iterrows():
                    shift = row[col]
                    if block in cfg.SHIFT_BLOCKS.get(shift, []):
                        total += 1
                        if row["skill"] == "RN":
                            rn += 1
                        elif row["skill"] == "PN":
                            pn += 1
                block_rows.append(
                    {
                        "day": day,
                        "block": block,
                        "total": total,
                        "RN": rn,
                        "PN": pn,
                    }
                )
        bc = pd.DataFrame(block_rows)
        bc["val"] = bc["total"].astype(str) + " (" + bc["RN"].astype(str) + ", " + bc["PN"].astype(str) + ")"
        bc.set_index(["block", "day"])["val"].unstack().to_excel(writer, sheet_name="BlockCoverage")

        shift_stats = roster_df[list(str(s) for s in range(1, args.days + 1))].stack().value_counts()
        shift_stats.to_excel(writer, sheet_name="ShiftStatistics")

        prefs_path = os.path.join(input_dir, "preferences.csv")
        prefs_df = pd.read_csv(prefs_path)
        request_stats, denied_rows = compute_request_stats(roster_df, prefs_df, args.days)
        request_stats.to_excel(writer, sheet_name="request summary")
        denied_rows.to_excel(writer, sheet_name="denied requests")

        print(shift_stats)
        print(request_stats)
        print(denied_rows)
        print(score)

    print(f"Roster written to {out_path}")


if __name__ == "__main__":
    main()
