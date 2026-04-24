"""
Step 3: Apply the WNBA cluster model to NCAA 2024-25 players.

NCAA CSV uses season totals and whole-number percentages — this script
normalizes them to per-game and decimal format before applying the model.

Inputs:
  data/Copy of WBB Data - ncaa_2425.csv
  data/cluster_model.pkl

Outputs:
  data/ncaa_archetype_assignments.csv
"""

import pandas as pd
import numpy as np
import pickle

ARCHETYPE_NAMES = {
    0: "Elite Two-Way Big",
    1: "Bench Guard/Wing",
    2: "High-Volume Wing/Guard",
    3: "Interior Specialist",
    4: "Efficient Role Player"
}

MIN_GAMES = 10

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

with open("data/cluster_model.pkl", "rb") as f:
    model = pickle.load(f)

scaler   = model["scaler"]
km       = model["kmeans"]
features = model["features"]

# ---------------------------------------------------------------------------
# Load and normalize NCAA data
# ---------------------------------------------------------------------------

ncaa = pd.read_csv("data/Copy of WBB Data - ncaa2526.csv")
print(f"Raw NCAA players: {len(ncaa)}")

# Convert all relevant columns to numeric
count_cols = ["G", "PTS", "STL", "BLK", "AST", "TOV", "3PA"]
pct_cols   = ["ORB%", "DRB%", "TRB%", "AST%", "USG%"]
ratio_cols = ["FG%", "3P%", "FT%", "eFG%", "TS%"]

for c in count_cols + pct_cols + ratio_cols:
    ncaa[c] = pd.to_numeric(ncaa[c], errors="coerce")

# Season totals → per game
for c in ["PTS", "STL", "BLK", "AST", "TOV", "3PA"]:
    ncaa[c] = ncaa[c] / ncaa["G"]

# Whole-number percentages → decimals
for c in pct_cols:
    ncaa[c] = ncaa[c] / 100

# Rename to match model feature names
col_map = {
    "FG%": "FG_PCT", "3P%": "FG3_PCT", "FT%": "FT_PCT",
    "eFG%": "EFG_PCT", "TS%": "TS_PCT", "ORB%": "OREB_PCT",
    "DRB%": "DREB_PCT", "TRB%": "REB_PCT", "AST%": "AST_PCT",
    "USG%": "USG_PCT", "3PA": "FG3A"
}
ncaa = ncaa.rename(columns=col_map)
ncaa["AST_TO"] = ncaa["AST"] / ncaa["TOV"].replace(0, np.nan)

# Filter minimum games
ncaa_eligible = ncaa[ncaa["G"] >= MIN_GAMES].dropna(subset=features).copy()
print(f"Eligible after filtering (min {MIN_GAMES} games): {len(ncaa_eligible)}")

# ---------------------------------------------------------------------------
# Assign archetypes
# ---------------------------------------------------------------------------

X = scaler.transform(ncaa_eligible[features].values)
ncaa_eligible["archetype_id"]   = km.predict(X)
ncaa_eligible["archetype_name"] = ncaa_eligible["archetype_id"].map(ARCHETYPE_NAMES)

print("\nArchetype distribution:")
for cid, name in ARCHETYPE_NAMES.items():
    n = (ncaa_eligible["archetype_id"] == cid).sum()
    print(f"  {cid} — {name}: {n}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

ncaa_eligible.to_csv("data/ncaa_archetype_assignments.csv", index=False)
print("\nSaved: ncaa_archetype_assignments.csv")
