"""
Step 4: Build salary projection model and generate the player lookup database.

Uses a Ridge regression trained on all WNBA players (stats + archetype) to
predict old-CBA salary, then applies the CBA conversion curve for new-CBA
projected salary. Each player gets a ±15% range around the median prediction.

Draft probability column is left as None — to be filled by the Stage 1
classifier (teammate's model).

Inputs:
  data/wnba_combined_2025.csv
  data/wnba_archetype_assignments.csv
  data/ncaa_archetype_assignments.csv
  data/cba_curve_params.json

Outputs:
  data/salary_model.pkl    — fitted salary regression model
  data/player_lookup.csv   — final lookup database
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import cross_val_score

FEATURES = [
    "PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "EFG_PCT", "TS_PCT",
    "OREB_PCT", "DREB_PCT", "REB_PCT", "AST_PCT", "AST_TO",
    "STL", "BLK", "USG_PCT", "FG3A"
]

ARCHETYPE_NAMES = {
    0: "Elite Two-Way Big",
    1: "Bench Guard/Wing",
    2: "High-Volume Wing/Guard",
    3: "Interior Specialist",
    4: "Efficient Role Player"
}

SALARY_BAND = 0.15  # ±15% around median prediction

# ---------------------------------------------------------------------------
# Load CBA curve
# ---------------------------------------------------------------------------

with open("data/cba_curve_params.json") as f:
    curve = json.load(f)

seg_params = [np.array(p) for p in curve["piecewise_params"]]
q33, q66 = curve["piecewise_splits"]

def power_law(x, a, b):
    return a * np.power(x, b)

def cba_convert(old_sal):
    old_sal = max(old_sal, 20000)
    if old_sal <= q33:   return power_law(old_sal, *seg_params[0])
    elif old_sal <= q66: return power_law(old_sal, *seg_params[1])
    else:                return power_law(old_sal, *seg_params[2])

# ---------------------------------------------------------------------------
# Build training set (WNBA players with salary)
# ---------------------------------------------------------------------------

wnba = pd.read_csv("data/wnba_combined_2025.csv")
assignments = pd.read_csv("data/wnba_archetype_assignments.csv")
wnba = pd.merge(wnba, assignments[["PLAYER_NAME","archetype_id"]], on="PLAYER_NAME", how="left")

for c in FEATURES + ["old_salary", "MIN"]:
    wnba[c] = pd.to_numeric(wnba[c], errors="coerce")
wnba["AST_TO"] = (pd.to_numeric(wnba["AST"], errors="coerce") /
                  pd.to_numeric(wnba["TOV"], errors="coerce").replace(0, np.nan))

df = wnba[wnba["MIN"] >= 10].dropna(subset=FEATURES + ["old_salary","archetype_id"]).copy()
print(f"Training samples: {len(df)}")

arch_dummies = pd.get_dummies(df["archetype_id"].astype(int), prefix="arch")
X_train = pd.concat([df[FEATURES].reset_index(drop=True),
                     arch_dummies.reset_index(drop=True)], axis=1).values
y_train = df["old_salary"].values

# ---------------------------------------------------------------------------
# Fit median quantile regression
# ---------------------------------------------------------------------------

model = QuantileRegressor(quantile=0.50, alpha=1.0, solver="highs")
model.fit(X_train, y_train)

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
print(f"CV R²: {scores.mean():.3f} ± {scores.std():.3f}")

# ---------------------------------------------------------------------------
# Apply to NCAA players
# ---------------------------------------------------------------------------

ncaa = pd.read_csv("data/ncaa_archetype_assignments.csv")
for c in FEATURES:
    ncaa[c] = pd.to_numeric(ncaa[c], errors="coerce")

ncaa_feat = ncaa.dropna(subset=FEATURES + ["archetype_id"]).copy().reset_index(drop=True)

ncaa_dummies = pd.get_dummies(ncaa_feat["archetype_id"].astype(int), prefix="arch")
for col in arch_dummies.columns:
    if col not in ncaa_dummies.columns:
        ncaa_dummies[col] = 0
ncaa_dummies = ncaa_dummies[arch_dummies.columns]

X_ncaa = pd.concat([ncaa_feat[FEATURES].reset_index(drop=True),
                    ncaa_dummies.reset_index(drop=True)], axis=1).values

pred_old = model.predict(X_ncaa).clip(min=20000)
ncaa_feat["salary_mid"]  = np.vectorize(cba_convert)(pred_old)
ncaa_feat["salary_low"]  = (ncaa_feat["salary_mid"] * (1 - SALARY_BAND)).round(0).astype(int)
ncaa_feat["salary_high"] = (ncaa_feat["salary_mid"] * (1 + SALARY_BAND)).round(0).astype(int)
ncaa_feat["salary_mid"]  = ncaa_feat["salary_mid"].round(0).astype(int)

# Deduplicate on player name
ncaa_dedup = ncaa_feat.sort_values("PTS", ascending=False).drop_duplicates("Player")
salary_df = ncaa_dedup.set_index("Player")[["salary_low","salary_mid","salary_high"]]

# ---------------------------------------------------------------------------
# Build lookup database
# ---------------------------------------------------------------------------

lookup = ncaa_feat[[
    "Player","Team","Class","Pos","G","PTS","FG_PCT","FG3_PCT","FT_PCT",
    "AST_PCT","REB_PCT","BLK","STL","USG_PCT","TS_PCT",
    "archetype_id","archetype_name"
]].drop_duplicates("Player").copy()

lookup = lookup.rename(columns={
    "Player": "player_name", "Team": "team", "Class": "class_year",
    "Pos": "position", "G": "games_played", "PTS": "pts_per_game",
    "FG_PCT": "fg_pct", "FG3_PCT": "fg3_pct", "FT_PCT": "ft_pct",
    "AST_PCT": "ast_pct", "REB_PCT": "reb_pct",
    "BLK": "blk_per_game", "STL": "stl_per_game",
    "USG_PCT": "usg_pct", "TS_PCT": "ts_pct",
})

for col in ["pts_per_game","blk_per_game","stl_per_game"]:
    lookup[col] = lookup[col].round(1)
for col in ["fg_pct","fg3_pct","ft_pct","ast_pct","reb_pct","usg_pct","ts_pct"]:
    lookup[col] = lookup[col].round(3)

lookup["draft_probability"] = None  # placeholder for Stage 1 classifier
lookup["proj_salary_low"]   = lookup["player_name"].map(salary_df["salary_low"])
lookup["proj_salary_mid"]   = lookup["player_name"].map(salary_df["salary_mid"])
lookup["proj_salary_high"]  = lookup["player_name"].map(salary_df["salary_high"])

lookup = lookup[[
    "player_name","team","class_year","position","games_played",
    "archetype_id","archetype_name",
    "draft_probability",
    "proj_salary_low","proj_salary_mid","proj_salary_high",
    "pts_per_game","fg_pct","fg3_pct","ft_pct",
    "ast_pct","reb_pct","blk_per_game","stl_per_game","usg_pct","ts_pct"
]]

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

with open("data/salary_model.pkl", "wb") as f:
    pickle.dump({"model": model, "features": FEATURES,
                 "arch_cols": arch_dummies.columns.tolist()}, f)

lookup.to_csv("data/player_lookup.csv", index=False)

print(f"\nLookup database: {len(lookup)} players")
print(f"Salary range: ${lookup['proj_salary_low'].min():,.0f} – ${lookup['proj_salary_high'].max():,.0f}")
print("Saved: salary_model.pkl, player_lookup.csv")
