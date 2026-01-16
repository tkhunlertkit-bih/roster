# solver.py

import argparse
import importlib
import os
from typing import Any, Dict, List

import pandas as pd
from ortools.sat.python import cp_model


def load_config(config_name: str) -> Any:
    return importlib.import_module(f"configs.{config_name}")


def load_coverage_blocks(path: str) -> Dict[int, Dict[str, Dict[str, int]]]:
    df = pd.read_csv(path)
    cov: Dict[int, Dict[str, Dict[str, int]]] = {}
    for _, row in df.iterrows():
        day = int(row["day"])
        block = str(row["block"])
        skill = str(row["skill"])
        req = int(row["required"])
        cov.setdefault(day, {}).setdefault(block, {})[skill] = req
    return cov


# (all helper functions unchanged except they now receive cfg and file paths)


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


def add_block_skill_coverage(model, x, nurses_df, days, coverage_blocks, cfg):
    """
    For each day, block and skill:
      sum over nurses with that skill of shifts covering the block >= required.
    Also enforces RN share <= 65% of (RN+PN) as a hard constraint per block.
    """
    S = len(cfg.SHIFT_NAMES)
    nurse_skills = list(nurses_df["skill"])
    num_nurses = len(nurse_skills)

    # which blocks each shift covers
    shift_blocks = {i: cfg.SHIFT_BLOCKS[cfg.SHIFT_NAMES[i]] for i in range(S)}

    for d in range(days):
        day = d + 1
        cov_day = coverage_blocks.get(day, {})

        for block in cfg.TIME_BLOCKS:
            cov_block = cov_day.get(block, {})

            # Collect terms for PN and RN in this block
            pn_terms = []
            rn_terms = []

            for n, skill_n in enumerate(nurse_skills):
                for s in range(S):
                    if block not in shift_blocks[s]:
                        continue
                    lit = x[n, d, s]
                    if skill_n == "PN":
                        pn_terms.append(lit)
                    elif skill_n == "RN":
                        rn_terms.append(lit)

            # Coverage constraints per skill (if required > 0)
            for skill in cfg.SKILLS:
                required = int(cov_block.get(skill, 0))
                if required == 0:
                    continue
                if skill == "PN":
                    model.Add(sum(pn_terms) >= required)
                elif skill == "RN":
                    model.Add(sum(rn_terms) >= required)

            # ---- RN ratio constraint: RN <= 65% of (RN+PN) ----
            pn_block = model.NewIntVar(0, num_nurses, f"pn_{day}_{block}")
            rn_block = model.NewIntVar(0, num_nurses, f"rn_{day}_{block}")
            tot_block = model.NewIntVar(0, num_nurses, f"tot_{day}_{block}")

            if pn_terms:
                model.Add(pn_block == sum(pn_terms))
            else:
                model.Add(pn_block == 0)

            if rn_terms:
                model.Add(rn_block == sum(rn_terms))
            else:
                model.Add(rn_block == 0)

            model.Add(tot_block == pn_block + rn_block)

            has_staff = model.NewBoolVar(f"has_staff_{day}_{block}")
            model.Add(tot_block > 0).OnlyEnforceIf(has_staff)
            model.Add(tot_block == 0).OnlyEnforceIf(has_staff.Not())

            # RN share <= 65% when there is at least one RN/PN
            # 100 * rn <= 65 * (rn + pn)
            model.Add(100 * rn_block <= 65 * tot_block).OnlyEnforceIf(has_staff)


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


def add_leave_constraints(model, x, nurses_df, prefs_df, days, cfg):
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
        pub_totals.append(pub_total)

    if cfg.EQUALIZE_PUB_COUNTS and N > 1:
        for i in range(1, N):
            model.Add(pub_totals[i] == pub_totals[0])


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
        for d in range(days - 3):
            off_sum = sum(x[n, d + k, s] for k in range(4) for s in off_idx)
            model.Add(off_sum <= cfg.MAX_OFF_IN_4_DAYS)

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
):
    """
    Objective:
      1) Minimize sum of excess hours (primary).
      2) Minimize preference penalties and their unfairness (secondary).
    """
    shift_index = {name: i for i, name in enumerate(cfg.SHIFT_NAMES)}
    N = len(nurses_df)

    penalties = []
    for n in range(N):
        terms = []
        for d in range(days):
            pref = prefs_df.loc[n, str(d + 1)]
            if pref in shift_index:
                s_pref = shift_index[pref]
                mismatch = model.NewBoolVar(f"mismatch_{n}_{d}")
                model.Add(x[n, d, s_pref] == 0).OnlyEnforceIf(mismatch)
                model.Add(x[n, d, s_pref] == 1).OnlyEnforceIf(mismatch.Not())
                terms.append(mismatch)
        total_penalty = model.NewIntVar(0, days * 2, f"penalty_{n}")
        if terms:
            model.Add(total_penalty == sum(terms))
        else:
            model.Add(total_penalty == 0)
        penalties.append(total_penalty)

    avg_penalty = model.NewIntVar(0, days * 2, "avg_penalty")
    model.Add(avg_penalty * N == sum(penalties))

    deviations = []
    for i, p in enumerate(penalties):
        dev = model.NewIntVar(0, days * 2, f"dev_{i}")
        model.Add(dev >= p - avg_penalty)
        model.Add(dev >= avg_penalty - p)
        deviations.append(dev)

    # Weights: hours more important than preferences
    W_HOURS = 10_000
    W_PREF = cfg.PREFERENCE_WEIGHT
    W_FAIR = cfg.FAIRNESS_WEIGHT

    model.Minimize(W_HOURS * sum(excess_hours_vars.values()) + W_PREF * sum(penalties) + W_FAIR * sum(deviations))


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


def solve_month(
    cfg,
    input_dir: str,
    month_days: int,
    max_time_sec: float = 60.0,
):
    nurses_path = os.path.join(input_dir, "nurses.csv")
    prefs_path = os.path.join(input_dir, "preferences.csv")
    coverage_blocks_path = os.path.join(input_dir, "coverage_blocks.csv")

    nurses_df = pd.read_csv(nurses_path)
    prefs_df = pd.read_csv(prefs_path)
    coverage_blocks = load_coverage_blocks(coverage_blocks_path)

    num_nurses = len(nurses_df)
    days = month_days

    model = cp_model.CpModel()
    x = create_variables(model, num_nurses, days, cfg)
    add_one_shift_per_day(model, x, num_nurses, days, cfg)
    add_long_off_vacation_rule(model, x, num_nurses, days, cfg)
    add_block_skill_coverage(model, x, nurses_df, days, coverage_blocks, cfg)
    total_hours_vars, excess_hours_vars = add_hours_constraints_and_excess(model, x, nurses_df, days, cfg)
    add_leave_constraints(model, x, nurses_df, prefs_df, days, cfg)
    add_forbidden_transitions(model, x, num_nurses, days, cfg)
    add_days_off_rules(model, x, num_nurses, days, cfg)
    add_preference_fair_objective(model, x, nurses_df, prefs_df, days, cfg, excess_hours_vars)

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
        help="Directory with nurses.csv, preferences.csv, coverage_blocks.csv",
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

    print(f"Roster written to {out_path}")


if __name__ == "__main__":
    main()
