# validate_roster.py

import importlib
from typing import Any

import pandas as pd


def load_config(config_name: str) -> Any:
    """
    Dynamically import a config module under configs/.
    Example: config_name='ipd_nurse' -> configs.ipd_nurse
    """
    return importlib.import_module(f"configs.{config_name}")


def validate_forbidden_transitions(roster_df: pd.DataFrame, cfg):
    """
    Check that no nurse has a forbidden shift->shift transition
    between consecutive days, according to cfg.FORBIDDEN_NEXT.
    """
    days = [c for c in roster_df.columns if c != "nurse_id"]
    for _, row in roster_df.iterrows():
        nurse = row["nurse_id"]
        for i in range(len(days) - 1):
            a = row[days[i]]
            b = row[days[i + 1]]
            if a in cfg.FORBIDDEN_NEXT and b in cfg.FORBIDDEN_NEXT[a]:
                raise AssertionError(f"{nurse} has forbidden transition {a}->{b} " f"on days {days[i]}-{days[i+1]}")


def validate_days_off_rules(roster_df: pd.DataFrame, cfg):
    """
    - Max cfg.MAX_OFF_IN_4_DAYS off days in any 4-day window.
    - After cfg.MAX_CONSEC_WORK_DAYS consecutive workdays,
      the next day must be off (unless it is beyond the horizon).
    """
    days = [c for c in roster_df.columns if c != "nurse_id"]
    D = len(days)

    for _, row in roster_df.iterrows():
        nurse = row["nurse_id"]
        off_flags = [row[d] in cfg.OFF_SHIFTS for d in days]

        # max off in any 4-day window
        for i in range(D - 3):
            if sum(off_flags[i : i + 4]) > cfg.MAX_OFF_IN_4_DAYS:
                raise AssertionError(
                    f"{nurse} has >{cfg.MAX_OFF_IN_4_DAYS} off days " f"in window {days[i]}-{days[i+3]}"
                )

        # after MAX_CONSEC_WORK_DAYS workdays -> next day off
        work_flags = [not f for f in off_flags]
        for i in range(D - cfg.MAX_CONSEC_WORK_DAYS):
            if all(work_flags[i : i + cfg.MAX_CONSEC_WORK_DAYS]):
                j = i + cfg.MAX_CONSEC_WORK_DAYS
                if j < D and not off_flags[j]:
                    raise AssertionError(
                        f"{nurse} works on {days[j]} after "
                        f"{cfg.MAX_CONSEC_WORK_DAYS} consecutive workdays "
                        f"({days[i]}-{days[j-1]})"
                    )


def validate_monthly_hours(roster_df: pd.DataFrame, nurses_df: pd.DataFrame, cfg):
    """
    Recompute total hours from roster_df using cfg.SHIFT_HOURS
    and check they are within [monthly_min_hours, monthly_max_hours]
    from nurses_df.
    """
    days = [c for c in roster_df.columns if c != "nurse_id"]
    merged = roster_df.merge(nurses_df, on="nurse_id")

    for _, row in merged.iterrows():
        nurse = row["nurse_id"]
        total = sum(cfg.SHIFT_HOURS[row[d]] for d in days)
        if not (row["monthly_min_hours"] <= total <= row["monthly_max_hours"]):
            raise AssertionError(
                f"{nurse} total hours {total} outside " f"[{row['monthly_min_hours']},{row['monthly_max_hours']}]"
            )
