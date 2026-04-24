"""
Step 1: Pull WNBA 2024-25 stats from official API, merge with salary files,
        and fit the old→new CBA salary conversion curve.

Outputs:
  data/wnba_combined_2025.csv   — WNBA stats + old salary + projected new salary
  data/cba_curve_params.json    — CBA conversion curve parameters
"""

import requests
import pandas as pd
import numpy as np
import unicodedata
import json
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_name(name):
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    return "".join(c for c in name if not unicodedata.combining(c))

def power_law(x, a, b):
    return a * np.power(x, b)

def fetch_wnba_endpoint(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://www.wnba.com",
        "Accept": "application/json"
    }
    r = requests.get(url, headers=headers, timeout=25)
    rs = r.json()["resultSets"][0]
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])

# ---------------------------------------------------------------------------
# 1. Pull WNBA stats from official API
# ---------------------------------------------------------------------------

print("Pulling WNBA base stats...")
base = fetch_wnba_endpoint(
    "https://stats.wnba.com/stats/leaguedashplayerstats?MeasureType=Base&PerMode=PerGame"
    "&Season=2024-25&SeasonType=Regular+Season&LeagueID=10&LastNGames=0&Month=0"
    "&OpponentTeamID=0&Period=0&PaceAdjust=N&PlusMinus=N&Rank=N&PORound=0"
)

print("Pulling WNBA advanced stats...")
adv = fetch_wnba_endpoint(
    "https://stats.wnba.com/stats/leaguedashplayerstats?MeasureType=Advanced&PerMode=PerGame"
    "&Season=2024-25&SeasonType=Regular+Season&LeagueID=10&LastNGames=0&Month=0"
    "&OpponentTeamID=0&Period=0&PaceAdjust=N&PlusMinus=N&Rank=N&PORound=0"
)

base_keep = ["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","AGE","GP","MIN",
             "PTS","FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT",
             "FTM","FTA","FT_PCT","OREB","DREB","REB","AST","STL","BLK","TOV"]
adv_keep  = ["PLAYER_ID","OFF_RATING","DEF_RATING","NET_RATING","AST_PCT","AST_TO",
             "AST_RATIO","OREB_PCT","DREB_PCT","REB_PCT","EFG_PCT","TS_PCT",
             "USG_PCT","PACE","PIE"]

stats = pd.merge(base[base_keep], adv[adv_keep], on="PLAYER_ID")
stats["player_norm"] = stats["PLAYER_NAME"].apply(normalize_name)
print(f"WNBA stats: {stats.shape}")

# ---------------------------------------------------------------------------
# 2. Build old-salary baseline from historical salary file
# ---------------------------------------------------------------------------

sal_old = pd.read_csv("data/Copy of WBB Data - wnba_salaries.csv")
sal_old["player_norm"] = sal_old["player_full_name"].apply(normalize_name)

year_cols = ["base_salary_2025","base_salary_2024","base_salary_2023",
             "base_salary_2022","base_salary_2021","base_salary_2020"]
sal_old["old_salary"] = sal_old[year_cols].bfill(axis=1).iloc[:, 0]
sal_old["old_salary_year"] = sal_old[year_cols].apply(
    lambda r: next((c.replace("base_salary_","") for c in year_cols if pd.notna(r[c])), None), axis=1
)
sal_old_dedup = (sal_old.dropna(subset=["old_salary"])
                        .sort_values("old_salary", ascending=False)
                        .drop_duplicates("player_norm"))

combined = pd.merge(stats, sal_old_dedup[["player_norm","player_full_name","team_abbrev",
                                           "old_salary","old_salary_year","signed_as"]],
                    on="player_norm", how="left")
print(f"Matched with old salary: {combined['old_salary'].notna().sum()} / {len(combined)}")

# ---------------------------------------------------------------------------
# 3. Fit old → new CBA conversion curve by signing type
# ---------------------------------------------------------------------------

sal_26 = pd.read_csv("data/Copy of WBB Data - wnba_salaries_2026.csv")
sal_26["player_norm"] = sal_26["player_full_name"].apply(normalize_name)

overlap = pd.merge(
    sal_old_dedup[["player_norm","old_salary"]],
    sal_26[["player_norm","base_salary_2026","signed_as"]].dropna(subset=["base_salary_2026"]),
    on="player_norm"
).dropna()

X, Y = overlap["old_salary"].values, overlap["base_salary_2026"].values

# Piecewise 3-segment fallback
q33, q66 = np.percentile(X, [33, 66])
seg_masks = [X <= q33, (X > q33) & (X <= q66), X > q66]
seg_params = []
for m in seg_masks:
    p, _ = curve_fit(power_law, X[m], Y[m], p0=[10, 0.8], maxfev=5000)
    seg_params.append(p.tolist())

# Per signing-type curves (min 4 samples)
type_params = {}
for stype, grp in overlap.groupby("signed_as"):
    if len(grp) >= 4:
        try:
            p, _ = curve_fit(power_law, grp["old_salary"].values, grp["base_salary_2026"].values,
                             p0=[10, 0.8], maxfev=5000)
            type_params[stype] = p.tolist()
        except Exception:
            pass

def predict_new_salary(old_sal, signed_as=None):
    if pd.isna(old_sal):
        return None
    if signed_as in type_params:
        return round(power_law(old_sal, *type_params[signed_as]), 0)
    if old_sal <= q33:   return round(power_law(old_sal, *seg_params[0]), 0)
    elif old_sal <= q66: return round(power_law(old_sal, *seg_params[1]), 0)
    else:                return round(power_law(old_sal, *seg_params[2]), 0)

combined["projected_new_salary"] = combined.apply(
    lambda r: predict_new_salary(r["old_salary"], r.get("signed_as")), axis=1
)

# Validation
overlap["pred"] = overlap.apply(lambda r: predict_new_salary(r["old_salary"], r["signed_as"]), axis=1)
err = np.median(np.abs(overlap["pred"] - overlap["base_salary_2026"]) / overlap["base_salary_2026"]) * 100
print(f"CBA curve median % error: {err:.1f}%")

# Save
curve_meta = {
    "type_params": type_params,
    "piecewise_splits": [q33, q66],
    "piecewise_params": seg_params,
    "median_pct_error": err
}
with open("data/cba_curve_params.json", "w") as f:
    json.dump(curve_meta, f, indent=2)

combined.to_csv("data/wnba_combined_2025.csv", index=False)
print("Saved: wnba_combined_2025.csv, cba_curve_params.json")
