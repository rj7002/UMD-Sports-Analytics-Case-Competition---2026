"""
Step 2: Cluster WNBA players into 5 archetypes using shared features
        (features that also exist in the NCAA CSV so the model transfers).

Inputs:
  data/wnba_combined_2025.csv

Outputs:
  data/cluster_model.pkl              — fitted scaler + KMeans model
  data/cluster_profiles.csv           — per-archetype stat + salary summary
  data/wnba_archetype_assignments.csv — each WNBA player's archetype
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit

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

K = 5
MIN_MINUTES = 10

# ---------------------------------------------------------------------------
# Load and prep
# ---------------------------------------------------------------------------

df = pd.read_csv("data/wnba_combined_2025.csv")

for c in FEATURES + ["MIN"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["AST_TO"] = (pd.to_numeric(df["AST"], errors="coerce") /
                pd.to_numeric(df["TOV"], errors="coerce").replace(0, np.nan))

df_cluster = df[df["MIN"] >= MIN_MINUTES].dropna(subset=FEATURES).copy()
print(f"Players eligible for clustering: {len(df_cluster)}")

# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[FEATURES].values)

km = KMeans(n_clusters=K, random_state=42, n_init=20)
df_cluster["archetype_id"] = km.fit_predict(X_scaled)
df_cluster["archetype_name"] = df_cluster["archetype_id"].map(ARCHETYPE_NAMES)

sil = silhouette_score(X_scaled, df_cluster["archetype_id"])
print(f"Silhouette score (k={K}): {sil:.4f}")

# ---------------------------------------------------------------------------
# Build archetype salary projections using CBA curve
# ---------------------------------------------------------------------------

with open("data/cba_curve_params.json") as f:
    curve = json.load(f)

type_params = {k: np.array(v) for k, v in curve["type_params"].items()}
q33, q66 = curve["piecewise_splits"]
seg_params = [np.array(p) for p in curve["piecewise_params"]]

def power_law(x, a, b):
    return a * np.power(x, b)

def project_salary(old_sal, signed_as="UFA"):
    if pd.isna(old_sal):
        return None
    if signed_as in type_params:
        return round(power_law(old_sal, *type_params[signed_as]), 0)
    if old_sal <= q33:   return round(power_law(old_sal, *seg_params[0]), 0)
    elif old_sal <= q66: return round(power_law(old_sal, *seg_params[1]), 0)
    else:                return round(power_law(old_sal, *seg_params[2]), 0)

# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

profiles = []
for cid in sorted(df_cluster["archetype_id"].unique()):
    grp = df_cluster[df_cluster["archetype_id"] == cid]
    sal_avg = grp["old_salary"].mean()
    sal_med = grp["old_salary"].median()
    profiles.append({
        "archetype_id": cid,
        "archetype_name": ARCHETYPE_NAMES[cid],
        "n": len(grp),
        "old_salary_avg": sal_avg,
        "old_salary_median": sal_med,
        "proj_new_salary_avg": project_salary(sal_avg, "UFA"),
        "proj_new_salary_median": project_salary(sal_med, "UFA"),
        **grp[FEATURES].mean().to_dict()
    })
    print(f"\n[{cid}] {ARCHETYPE_NAMES[cid]}  n={len(grp)}  old_salary_avg=${sal_avg:,.0f}")
    p = grp[FEATURES].mean()
    print(f"  PTS={p['PTS']:.1f}  TS%={p['TS_PCT']:.3f}  USG%={p['USG_PCT']:.3f}  "
          f"AST%={p['AST_PCT']:.3f}  BLK={p['BLK']:.2f}  3PA={p['FG3A']:.1f}")
    print(f"  {', '.join(grp.sort_values('old_salary', ascending=False)['PLAYER_NAME'].head(5).tolist())}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

with open("data/cluster_model.pkl", "wb") as f:
    pickle.dump({"scaler": scaler, "kmeans": km, "features": FEATURES, "k": K})

pd.DataFrame(profiles).to_csv("data/cluster_profiles.csv", index=False)

df_cluster[["PLAYER_NAME","TEAM_ABBREVIATION","archetype_id","archetype_name",
            "old_salary","old_salary_year","signed_as"] + FEATURES].to_csv(
    "data/wnba_archetype_assignments.csv", index=False
)

print("\nSaved: cluster_model.pkl, cluster_profiles.csv, wnba_archetype_assignments.csv")
