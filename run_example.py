# run_example.py

import argparse
import os

import pandas as pd

from generate_mock_data import (build_coverage_from_beds_rn_pn,
                                generate_mock_nurses,
                                generate_mock_preferences)
from generate_mock_data import load_config as load_cfg_data
from solver import load_config as load_cfg_solver
from solver import solve_month
from validate_roster import load_config as load_cfg_val
from validate_roster import (validate_days_off_rules,
                             validate_forbidden_transitions,
                             validate_monthly_hours)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ipd_nurse")
    parser.add_argument("--days", type=int, default=31)
    parser.add_argument("--nurses", type=int, default=16)
    args = parser.parse_args()

    cfg_data = load_cfg_data(args.config)
    input_dir = os.path.join("mock_data", args.config)

    # 1) mock inputs
    generate_mock_nurses(cfg_data, out_dir=input_dir, num_nurses=args.nurses)
    generate_mock_preferences(cfg_data, out_dir=input_dir, month_days=args.days)
    build_coverage_from_beds_rn_pn(cfg_data, out_dir=input_dir, beds_days=args.days)

    # 2) solve
    cfg_solver = load_cfg_solver(args.config)
    roster_df = solve_month(
        cfg_solver,
        input_dir=input_dir,
        month_days=args.days,
    )
    out_path = os.path.join(input_dir, "roster_output.xlsx")
    roster_df.to_excel(out_path, index=False)
    print(f"Roster written to {out_path}")

    # 3) validate
    cfg_val = load_cfg_val(args.config)
    nurses_df = pd.read_csv(os.path.join(input_dir, "nurses.csv"))
    validate_forbidden_transitions(roster_df, cfg_val)
    validate_days_off_rules(roster_df, cfg_val)
    validate_monthly_hours(roster_df, nurses_df, cfg_val)
    print("Validation passed.")


if __name__ == "__main__":
    main()
