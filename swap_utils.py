import logging

import pandas as pd


def has_forbidden_transitions_for_nurse(roster_df, cfg, nurse_idx, days):
    row = roster_df.iloc[nurse_idx]
    forbidden_next = cfg.FORBIDDEN_NEXT

    for d in range(1, days):
        s_today = str(row[str(d)])
        s_next = str(row[str(d + 1)])
        if s_today in forbidden_next and s_next in forbidden_next[s_today]:
            return True
    return False


def request_delta_for_swap(roster_df, prefs_df, cfg, day, i, j):
    """
    Compute change in satisfied requests if we swap nurse i and j on given day.
    Positive delta = improvement.
    Only uses shift prefs and X/R, not fairness.
    """
    col = str(day)
    row_i = roster_df.iloc[i]
    row_j = roster_df.iloc[j]

    nid_i = row_i["nurse_id"]
    nid_j = row_j["nurse_id"]

    s_i = str(row_i[col])
    s_j = str(row_j[col])

    # do not change work/off status
    off_set = set(cfg.OFF_SHIFTS)
    if (s_i in off_set) != (s_j in off_set):
        return -1e9  # invalid swap

    pref_i = prefs_df.loc[prefs_df["nurse_id"] == nid_i, col].iloc[0]
    pref_j = prefs_df.loc[prefs_df["nurse_id"] == nid_j, col].iloc[0]

    def pref_score(pref, assigned):
        if pd.isna(pref) or pref == "":
            return 0
        pref = str(pref)
        if pref == "R":
            pref = "X"

        if pref == "X":
            # wants day off
            return 1 if assigned in off_set else 0
        else:
            return 1 if assigned == pref else 0

    # current total satisfied on this day for these two
    cur = pref_score(pref_i, s_i) + pref_score(pref_j, s_j)
    # after swap: i gets s_j, j gets s_i
    new = pref_score(pref_i, s_j) + pref_score(pref_j, s_i)

    return new - cur


def nurse_pref_score(roster_df, prefs_df, cfg, nurse_idx, days):
    row = roster_df.iloc[nurse_idx]
    nid = row["nurse_id"]
    off_set = set(cfg.OFF_SHIFTS)

    prefs_n = prefs_df[prefs_df["nurse_id"] == nid].iloc[0]

    score = 0
    for d in range(1, days + 1):
        col = str(d)
        pref = prefs_n[col]
        assg = str(row[col])

        if pd.isna(pref) or pref == "":
            continue
        pref = str(pref)
        if pref == "R":
            pref = "X"

        if pref == "X":
            if assg in off_set:
                score += 1
        else:
            if assg == pref:
                score += 1
    return score


def improve_by_month_swaps(roster_df, nurses_df, prefs_df, cfg, days, max_passes=10):
    """
    Swap entire monthly patterns between nurses with same skill
    when this increases total preference score (sum over month).
    """
    skill_map = dict(zip(nurses_df["nurse_id"], nurses_df["skill"]))

    for round in range(max_passes):
        improved_any = False

        # Precompute nurse index by id
        idx_by_id = {nid: idx for idx, nid in enumerate(roster_df["nurse_id"])}

        nurse_ids = list(roster_df["nurse_id"])
        nN = len(nurse_ids)

        for a in range(nN):
            nid_a = nurse_ids[a]
            skill_a = skill_map[nid_a]
            for b in range(a + 1, nN):
                nid_b = nurse_ids[b]
                if skill_map[nid_b] != skill_a:
                    continue  # keep skill mix

                idx_a = idx_by_id[nid_a]
                idx_b = idx_by_id[nid_b]

                # current total score for these two
                cur = nurse_pref_score(roster_df, prefs_df, cfg, idx_a, days) + nurse_pref_score(
                    roster_df, prefs_df, cfg, idx_b, days
                )

                # simulate swap
                row_a = roster_df.loc[idx_a, :]
                row_b = roster_df.loc[idx_b, :]

                # swap only the day columns 1..days
                day_cols = [str(d) for d in range(1, days + 1)]
                tmp_a = row_a[day_cols].copy()
                tmp_b = row_b[day_cols].copy()

                roster_df.loc[idx_a, day_cols] = tmp_b.values
                roster_df.loc[idx_b, day_cols] = tmp_a.values

                new = nurse_pref_score(roster_df, prefs_df, cfg, idx_a, days) + nurse_pref_score(
                    roster_df, prefs_df, cfg, idx_b, days
                )

                a_valid = not has_forbidden_transitions_for_nurse(roster_df, cfg, idx_a, days)
                b_valid = not has_forbidden_transitions_for_nurse(roster_df, cfg, idx_b, days)
                if (new > cur) and a_valid and b_valid:
                    logging.info(f"swapping the whole month round {round} ::{nid_a} <--> {nid_b}")
                    improved_any = True  # keep swap
                else:
                    # revert
                    roster_df.loc[idx_a, day_cols] = tmp_a.values
                    roster_df.loc[idx_b, day_cols] = tmp_b.values

        if not improved_any:
            break

    return roster_df


def improve_by_day_swaps(roster_df, nurses_df, prefs_df, cfg, days, max_passes=5):
    skill_map = dict(zip(nurses_df["nurse_id"], nurses_df["skill"]))
    off_set = set(cfg.OFF_SHIFTS)

    for _ in range(max_passes):
        improved_any = False

        for d in range(1, days + 1):
            col = str(d)

            # group by skill
            indices_by_skill = {}
            for idx, row in roster_df.iterrows():
                nid = row["nurse_id"]
                skill = skill_map[nid]
                indices_by_skill.setdefault(skill, []).append(idx)

            for skill, idx_list in indices_by_skill.items():
                for i_pos in range(len(idx_list)):
                    i = idx_list[i_pos]
                    s_i = roster_df.loc[i, col]
                    for j_pos in range(i_pos + 1, len(idx_list)):
                        j = idx_list[j_pos]
                        s_j = roster_df.loc[j, col]

                        # keep work/off
                        if (s_i in off_set) != (s_j in off_set):
                            continue

                        delta = request_delta_for_swap(roster_df, prefs_df, cfg, d, i, j)
                        if delta > 0:
                            roster_df.loc[i, col], roster_df.loc[j, col] = s_j, s_i
                            ok_i = not has_forbidden_transitions_for_nurse(roster_df, cfg, i, days)
                            ok_j = not has_forbidden_transitions_for_nurse(roster_df, cfg, j, days)
                            if ok_i and ok_j:
                                improved_any = True
                            else:
                                # revert
                                roster_df.loc[i, col], roster_df.loc[j, col] = s_i, s_j

        if not improved_any:
            break

    return roster_df
