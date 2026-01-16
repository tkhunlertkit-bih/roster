# generate_mock_data.py

import argparse
import importlib
import os
import random
from typing import Any, Dict

import pandas as pd


def load_config(config_name: str) -> Any:
    return importlib.import_module(f"configs.{config_name}")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_mock_nurses(config, out_dir: str, num_nurses=16, seed=0):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "nurses.csv")
    random.seed(seed)
    nurses = []
    for i in range(num_nurses):
        nurse_id = f"N{i+1:02d}"
        level = random.randint(1, 4)
        role = "RN_IPD"
        skill = random.choices(config.SKILLS, weights=config.SKILL_PROPORTION, k=1)[0]
        phone = f"08{i:08d}"  # dummy phone number
        nurses.append(
            {
                "nurse_id": nurse_id,
                "name": f"Nurse{i+1}",
                "phone_number": phone,
                "level": level,
                "role": role,
                "skill": skill,
                "monthly_min_hours": 176,
                "monthly_max_hours": 216,
                "is_fixed_night": 0,
            }
        )
    pd.DataFrame(nurses).to_csv(path, index=False)
    print(f"Wrote {path}")


def generate_mock_preferences(
    config,
    out_dir: str,
    month_days=31,
    seed=0,
):
    ensure_dir(out_dir)
    nurses_path = os.path.join(out_dir, "nurses.csv")
    path = os.path.join(out_dir, "preferences.csv")

    random.seed(seed)
    nurses_df = pd.read_csv(nurses_path)
    nurse_ids = list(nurses_df["nurse_id"])

    data: Dict[str, list] = {"nurse_id": nurse_ids}
    working_shifts = list(set(config.SHIFT_NAMES) - set(config.OFF_SHIFTS))
    for d in range(1, month_days + 1):
        col = []
        for _ in nurse_ids:
            r = random.random()
            if r < 0.01:
                col.append("BD")
            elif r < 0.03:
                col.append("PUB")
            elif r < 0.13:
                col.append("X")  # use to be "V" but nurses would only submit X, excesses would be V.
            elif r < 0.20:
                col.append(random.choice(working_shifts))
            else:  # no preferences.
                col.append("")
        data[str(d)] = col

    pd.DataFrame(data).to_csv(path, index=False)
    print(f"Wrote {path}")


def build_coverage_from_beds_rn_pn(
    config,
    out_dir: str,
    beds_days: int,
):
    """
    Creates mock:
      beds_per_day.csv
      beds_lookup_blocks.csv
      coverage_blocks.csv
    inside out_dir.
    """
    ensure_dir(out_dir)

    beds_path = os.path.join(out_dir, "beds_per_day.csv")
    lookup_path = os.path.join(out_dir, "beds_lookup_blocks.csv")
    coverage_path = os.path.join(out_dir, "coverage_blocks.csv")

    beds_per_day = pd.DataFrame(
        {
            "day": list(range(1, beds_days + 1)),
            "beds": [10] * beds_days,
        }
    )
    beds_per_day.to_csv(beds_path, index=False)

    lookup = pd.DataFrame(
        {
            "beds": [10],
            "B1_RN": [2],
            "B1_PN": [2],
            "B2_RN": [2],
            "B2_PN": [2],
            "B3_RN": [2],
            "B3_PN": [2],
            "B4_RN": [2],
            "B4_PN": [1],
        }
    )
    lookup.to_csv(lookup_path, index=False)

    # build coverage_blocks.csv
    beds_df = pd.read_csv(beds_path)
    lookup_df = pd.read_csv(lookup_path)

    rows = []
    for _, row in beds_df.iterrows():
        day = int(row["day"])
        beds = int(row["beds"])
        match = lookup_df.loc[lookup_df["beds"] == beds]
        if match.empty:
            raise ValueError(f"No lookup entry for beds={beds}")
        m = match.iloc[0]

        for b in config.TIME_BLOCKS:
            for skill in config.SKILLS:
                col = f"{b}_{skill}"
                required = int(m[col])
                rows.append(
                    {
                        "day": day,
                        "block": b,
                        "skill": skill,
                        "required": required,
                    }
                )

    pd.DataFrame(rows).to_csv(coverage_path, index=False)
    print(f"Wrote {coverage_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ipd_nurse")
    parser.add_argument("--days", type=int, default=31)
    parser.add_argument("--nurses", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = os.path.join("mock_data", args.config)

    generate_mock_nurses(cfg, out_dir=out_dir, num_nurses=args.nurses)
    generate_mock_preferences(cfg, out_dir=out_dir, month_days=args.days)
    build_coverage_from_beds_rn_pn(cfg, out_dir=out_dir, beds_days=args.days)


if __name__ == "__main__":
    main()
