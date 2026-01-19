# solver.py

import argparse
import importlib
import os
from typing import Any, Dict, List

import pandas as pd
from ortools.sat.python import cp_model

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


def add_preference_fair_objective(
    model,
    x,
    nurses_df,
    prefs_df,
    days,
    cfg,
    excess_hours_vars,
    total_hours_all,
    beds_sum,
    fte_uos_threshold,
):
    """
    Objective:
      1) Minimize sum of excess hours (primary).
      2) Minimize preference penalties and their unfairness (secondary).
    """
    shift_index = {name: i for i, name in enumerate(cfg.SHIFT_NAMES)}
    N = len(nurses_df)

    penalties = []
    dayoff_penalties = []
    long_shift_terms = []
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

            # requested day off
            if pref == "X":
                # any non-off assignment is a strong violation
                is_off = model.NewBoolVar(f"is_off_{n}_{d}")
                # off if assigned in any OFF_SHIFTS
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

                    # reward matches (so that pref shifts win over generic penalties)
                    match = model.NewBoolVar(f"match_{n}_{d}")
                    model.Add(x[n, d, s_pref] == 1).OnlyEnforceIf(match)
                    model.Add(x[n, d, s_pref] == 0).OnlyEnforceIf(match.Not())
                    # store as negative penalty later by substracting from objective via a weight
                    long_shift_terms.append((match, pref))

                    # If preferred shift is a LONG shift, track a matched variables
                    if pref in cfg.LONG_SHIFTS:
                        match_long = model.NewBoolVar(f"match_long_{n}_{d}")
                        model.Add(x[n, d, s_pref] == 1).OnlyEnforceIf(match_long)
                        model.Add(x[n, d, s_pref] == 0).OnlyEnforceIf(match_long.Not())
                        long_pref_matches.append(match_long)

        # aggregate
        total_penalty = model.NewIntVar(0, days * 2, f"penalty_{n}")
        model.Add(total_penalty == sum(terms)) if terms else model.Add(total_penalty == 0)
        penalties.append(total_penalty)

        if has_pref:
            requesters.append(n)

        dayoff_penalty = model.NewIntVar(0, days, f"dayoff_penalty_{n}")
        model.Add(dayoff_penalty == sum(dayoff_terms)) if dayoff_terms else model.Add(dayoff_penalty == 0)
        dayoff_penalties.append(dayoff_penalty)

    long_penalty_terms = []
    for n in range(N):
        for d in range(days):
            is_long = model.NewBoolVar(f"is_long_{n}_{d}")
            long_lits = []
            for s_name in cfg.LONG_SHIFTS:
                if s_name in shift_index:
                    long_lits.append(x[n, d, shift_index[s_name]])
            if long_lits:
                long_sum = model.NewIntVar(0, len(long_lits), f"long_sum_{n}_{d}")
                model.Add(long_sum == sum(long_lits))
                model.Add(long_sum >= 1).OnlyEnforceIf(is_long)
                model.Add(long_sum == 0).OnlyEnforceIf(is_long.Not())
                long_penalty_terms.append(is_long)

    avg_penalty = model.NewIntVar(0, days * 2, "avg_penalty")
    model.Add(avg_penalty * N == sum(penalties))

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

    # FTE/UOS soft penalty_
    # target_max_hours = fte
    target_max_hours = int(fte_uos_threshold * beds_sum * 8)
    over_fte = model.NewIntVar(0, 2 * target_max_hours, "over_fte")
    model.Add(over_fte >= total_hours_all - target_max_hours)
    model.Add(over_fte >= 0)

    # Weights: hours more important than preferences
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
        + W_PREF * sum(penalties)  # penalize mismatches
        + W_FAIR * sum(deviations)  # fairness of prefs
        + W_LONG * sum(long_penalty_terms)  # generic long-shift penalty
        - W_LONG_PREF * sum(long_pref_matches)  # reward long shift that matches preferences
        + W_FTE * over_fte
    )


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


def solve_month(
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
    return roster_df


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

    roster_df = solve_month(
        cfg,
        input_dir=input_dir,
        month_days=args.days,
        pub_days_per_nurse=args.pub_days_per_nurse,
        fte_uos_threshold=args.fte_uos_threshold,
        max_time_sec=args.max_time,
    )

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

    print(f"Roster written to {out_path}")


if __name__ == "__main__":
    main()
