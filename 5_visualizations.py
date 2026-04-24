"""
Step 5: Generate all presentation visualizations.

Outputs (saved to data/viz/):
  01_archetype_clusters_pca.png
  02_cba_salary_curve.png
  03_archetype_salary_ranges.png
  04_ncaa_archetype_distribution.png
  05_top_prospects_by_archetype.png
  06_cluster_selection.png
  07_player_lookup_showcase.png
  08_wnba_salary_by_archetype.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import unicodedata
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import os
os.makedirs("data/viz", exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
COLORS = {
    0: "#E63946",   # Elite Two-Way Big        — red
    1: "#457B9D",   # Bench Guard/Wing          — steel blue
    2: "#F4A261",   # High-Volume Wing/Guard    — orange
    3: "#2A9D8F",   # Interior Specialist       — teal
    4: "#9B5DE5",   # Efficient Role Player     — purple
}
ARCHETYPE_NAMES = {
    0: "Elite Two-Way Big",
    1: "Bench Guard/Wing",
    2: "High-Volume Wing/Guard",
    3: "Interior Specialist",
    4: "Efficient Role Player",
}
FEATURES = [
    "PTS","FG_PCT","FG3_PCT","FT_PCT","EFG_PCT","TS_PCT",
    "OREB_PCT","DREB_PCT","REB_PCT","AST_PCT","AST_TO",
    "STL","BLK","USG_PCT","FG3A"
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

def save(fig, name):
    fig.savefig(f"data/viz/{name}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: data/viz/{name}")

def normalize_name(name):
    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name)
    return "".join(c for c in name if not unicodedata.combining(c))

# ── Load data ─────────────────────────────────────────────────────────────────
wnba  = pd.read_csv("data/wnba_archetype_assignments.csv")
ncaa  = pd.read_csv("data/ncaa_archetype_assignments.csv")
lookup= pd.read_csv("data/player_lookup.csv")
profs = pd.read_csv("data/cluster_profiles.csv")

with open("data/cba_curve_params.json") as f:
    curve = json.load(f)

for c in FEATURES + ["old_salary"]:
    wnba[c] = pd.to_numeric(wnba[c], errors="coerce")

# ─────────────────────────────────────────────────────────────────────────────
# 01 · Archetype clusters — PCA scatter (WNBA)
# ─────────────────────────────────────────────────────────────────────────────
print("01 — Archetype clusters PCA...")
df_pca = wnba.dropna(subset=FEATURES + ["archetype_id"]).copy()
X_sc = StandardScaler().fit_transform(df_pca[FEATURES].values)
coords = PCA(n_components=2, random_state=42).fit_transform(X_sc)
df_pca["pc1"], df_pca["pc2"] = coords[:,0], coords[:,1]

fig, ax = plt.subplots(figsize=(10, 7))
for cid, name in ARCHETYPE_NAMES.items():
    sub = df_pca[df_pca["archetype_id"] == cid]
    ax.scatter(sub["pc1"], sub["pc2"], c=COLORS[cid], label=name,
               s=80, alpha=0.85, edgecolors="white", linewidths=0.5)

# Label notable players
notables = ["A'ja Wilson","Caitlin Clark","Breanna Stewart","Napheesa Collier",
            "Arike Ogunbowale","Sabrina Ionescu","Jonquel Jones","Angel Reese"]
for _, row in df_pca[df_pca["PLAYER_NAME"].isin(notables)].iterrows():
    ax.annotate(row["PLAYER_NAME"].split()[-1], (row["pc1"], row["pc2"]),
                fontsize=7.5, fontweight="bold",
                xytext=(5, 4), textcoords="offset points")

ax.set_title("WNBA Player Archetypes (PCA)", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Principal Component 1", fontsize=11)
ax.set_ylabel("Principal Component 2", fontsize=11)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.2)
save(fig, "01_archetype_clusters_pca.png")

# ─────────────────────────────────────────────────────────────────────────────
# 02 · CBA salary conversion curve
# ─────────────────────────────────────────────────────────────────────────────
print("02 — CBA salary curve...")
sal_old = pd.read_csv("data/Copy of WBB Data - wnba_salaries.csv")
sal_26  = pd.read_csv("data/Copy of WBB Data - wnba_salaries_2026.csv")
sal_old["player_norm"] = sal_old["player_full_name"].apply(normalize_name)
sal_26["player_norm"]  = sal_26["player_full_name"].apply(normalize_name)

year_cols = ["base_salary_2025","base_salary_2024","base_salary_2023",
             "base_salary_2022","base_salary_2021","base_salary_2020"]
sal_old["old_salary"] = sal_old[year_cols].bfill(axis=1).iloc[:,0]
sal_old_d = sal_old.dropna(subset=["old_salary"]).sort_values("old_salary",ascending=False).drop_duplicates("player_norm")

overlap = pd.merge(sal_old_d[["player_norm","old_salary"]],
                   sal_26[["player_norm","base_salary_2026","signed_as"]].dropna(subset=["base_salary_2026"]),
                   on="player_norm").dropna()

type_params = {k: np.array(v) for k,v in curve["type_params"].items()}
seg_params  = [np.array(p) for p in curve["piecewise_params"]]
q33, q66    = curve["piecewise_splits"]

def power_law(x, a, b): return a * np.power(x, b)
def cba_convert(old):
    old = max(old, 20000)
    if old <= q33:   return power_law(old, *seg_params[0])
    elif old <= q66: return power_law(old, *seg_params[1])
    else:            return power_law(old, *seg_params[2])

x_line = np.linspace(overlap["old_salary"].min(), overlap["old_salary"].max(), 300)
y_line = np.vectorize(cba_convert)(x_line)

fig, ax = plt.subplots(figsize=(10, 6))
type_colors = {"UFA":"#457B9D","Rookie":"#2A9D8F","RFA":"#F4A261",
               "Core":"#E63946","Reserved":"#9B5DE5","SuspCE":"#aaa"}
for stype, grp in overlap.groupby("signed_as"):
    ax.scatter(grp["old_salary"], grp["base_salary_2026"],
               c=type_colors.get(stype,"#888"), label=stype,
               s=60, alpha=0.8, edgecolors="white", linewidths=0.4)

ax.plot(x_line, y_line, color="#222", lw=2.5, ls="--", label="Fitted curve", zorder=5)
ax.set_title("Old CBA → New CBA Salary Conversion", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Old CBA Salary ($)", fontsize=11)
ax.set_ylabel("New CBA Salary — 2026 ($)", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}k"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}k"))
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.2)
save(fig, "02_cba_salary_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# 03 · Archetype projected salary ranges
# ─────────────────────────────────────────────────────────────────────────────
print("03 — Archetype salary ranges...")
order = profs.sort_values("proj_new_salary_avg", ascending=True)
names = [ARCHETYPE_NAMES[i] for i in order["archetype_id"]]
mids  = order["proj_new_salary_avg"].values
lows  = mids * 0.85
highs = mids * 1.15
cols  = [COLORS[i] for i in order["archetype_id"]]

fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(names))
for i, (lo, mid, hi, c) in enumerate(zip(lows, mids, highs, cols)):
    ax.barh(i, hi - lo, left=lo, height=0.5, color=c, alpha=0.35)
    ax.plot([mid, mid], [i - 0.25, i + 0.25], color=c, lw=3)
    ax.text(hi + 10000, i, f"${mid/1e3:.0f}k", va="center", fontsize=9, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=10)
ax.set_title("Projected New CBA Salary by Archetype", fontsize=15, fontweight="bold", pad=12)
ax.set_xlabel("Projected Salary ($)", fontsize=11)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}k"))
ax.grid(True, axis="x", alpha=0.2)
save(fig, "03_archetype_salary_ranges.png")

# ─────────────────────────────────────────────────────────────────────────────
# 04 · NCAA archetype distribution
# ─────────────────────────────────────────────────────────────────────────────
print("04 — NCAA archetype distribution...")
counts = ncaa["archetype_id"].value_counts().sort_index()
names_list = [ARCHETYPE_NAMES[i] for i in counts.index]
colors_list = [COLORS[i] for i in counts.index]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart
bars = ax1.bar(range(len(counts)), counts.values, color=colors_list, edgecolor="white", linewidth=0.8)
ax1.set_xticks(range(len(counts)))
ax1.set_xticklabels(names_list, rotation=20, ha="right", fontsize=9)
ax1.set_ylabel("Number of Players", fontsize=11)
ax1.set_title("NCAA Players per Archetype", fontsize=13, fontweight="bold")
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             str(val), ha="center", fontsize=10, fontweight="bold")
ax1.grid(True, axis="y", alpha=0.2)

# Pie chart
wedges, texts, autotexts = ax2.pie(
    counts.values, labels=names_list, colors=colors_list,
    autopct="%1.1f%%", startangle=140,
    textprops={"fontsize": 8.5}, pctdistance=0.78
)
for at in autotexts:
    at.set_fontweight("bold")
ax2.set_title("Archetype Share", fontsize=13, fontweight="bold")

fig.suptitle("2024-25 NCAA Women's Basketball — Archetype Distribution\n(3,706 players, min 10 games)",
             fontsize=13, y=1.02)
save(fig, "04_ncaa_archetype_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 05 · Top 5 prospects per archetype
# ─────────────────────────────────────────────────────────────────────────────
print("05 — Top prospects by archetype...")
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
fig.suptitle("Top 5 NCAA Prospects per Archetype (by PTS/g)", fontsize=14, fontweight="bold", y=1.02)

for ax, (cid, name) in zip(axes, ARCHETYPE_NAMES.items()):
    grp = lookup[lookup["archetype_id"] == cid].nlargest(5, "pts_per_game")
    player_labels = [p.split()[-1] for p in grp["player_name"]]
    mids  = grp["proj_salary_mid"].values
    lows  = grp["proj_salary_low"].values
    highs = grp["proj_salary_high"].values
    y = np.arange(len(player_labels))

    ax.barh(y, highs - lows, left=lows, height=0.5, color=COLORS[cid], alpha=0.35)
    ax.plot(mids, y, "o", color=COLORS[cid], ms=7, zorder=5)
    for i, (lo, mid, hi) in enumerate(zip(lows, mids, highs)):
        ax.plot([lo, hi], [i, i], color=COLORS[cid], lw=2)

    ax.set_yticks(y)
    ax.set_yticklabels(player_labels, fontsize=8.5)
    ax.set_title(name, fontsize=9, fontweight="bold", color=COLORS[cid], pad=6)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}k"))
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, axis="x", alpha=0.2)
    ax.invert_yaxis()

plt.tight_layout()
save(fig, "05_top_prospects_by_archetype.png")

# ─────────────────────────────────────────────────────────────────────────────
# 06 · Cluster selection (elbow + silhouette) — replot cleanly
# ─────────────────────────────────────────────────────────────────────────────
print("06 — Cluster selection...")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df_c = wnba.dropna(subset=FEATURES + ["archetype_id"]).copy()
X_sc2 = StandardScaler().fit_transform(df_c[FEATURES].values)
ks, inertias, sils = range(2, 11), [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_sc2)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_sc2, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(ks), inertias, "o-", color="#457B9D", lw=2, ms=8)
ax1.axvline(5, color="#E63946", ls="--", lw=1.5, label="Chosen k=5")
ax1.set_xlabel("Number of Clusters (k)"); ax1.set_ylabel("Inertia (WCSS)")
ax1.set_title("Elbow Method", fontsize=13, fontweight="bold")
ax1.set_xticks(list(ks)); ax1.grid(True, alpha=0.2); ax1.legend()

ax2.plot(list(ks), sils, "o-", color="#2A9D8F", lw=2, ms=8)
ax2.axvline(5, color="#E63946", ls="--", lw=1.5, label="Chosen k=5")
ax2.set_xlabel("Number of Clusters (k)"); ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score", fontsize=13, fontweight="bold")
ax2.set_xticks(list(ks)); ax2.grid(True, alpha=0.2); ax2.legend()

fig.suptitle("Choosing the Number of Archetypes", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
save(fig, "06_cluster_selection.png")

# ─────────────────────────────────────────────────────────────────────────────
# 07 · Player lookup showcase (table)
# ─────────────────────────────────────────────────────────────────────────────
print("07 — Player lookup showcase...")
showcase_names = [
    "Paige Bueckers","Ta'Niya Latson","JuJu Watkins","Audi Crooks",
    "Hannah Hidalgo","Sonia Citron","Kiki Iriafen","Aneesah Morrow",
    "Saniya Rivers","Sarah Ashlee Barker"
]
rows = []
for name in showcase_names:
    first, last = name.split()[0], name.split()[-1]
    match = lookup[lookup["player_name"].str.contains(first, case=False, na=False) &
                   lookup["player_name"].str.contains(last,  case=False, na=False)]
    if len(match):
        r = match.iloc[0]
        rows.append({
            "Player": r["player_name"],
            "Team": r["team"].replace(" Women's","").replace(" Huskies","").replace(" Seminoles","")
                              .replace(" Trojans","").replace(" Cyclones","").replace(" Irish","")
                              .replace(" Tar Heels","").replace(" Gamecocks","").replace(" Wolfpack","")
                              .replace(" Crimson",""),
            "Class": r["class_year"],
            "Archetype": r["archetype_name"],
            "PTS/g": f"{r['pts_per_game']:.1f}",
            "TS%": f"{r['ts_pct']:.3f}",
            "Salary Range": f"${r['proj_salary_low']/1e3:.0f}k – ${r['proj_salary_high']/1e3:.0f}k"
        })

df_show = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")
tbl = ax.table(
    cellText=df_show.values,
    colLabels=df_show.columns,
    cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.6)

# Style header
for j in range(len(df_show.columns)):
    tbl[0, j].set_facecolor("#222831")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Color rows by archetype
arch_col_map = {name: COLORS[cid] for cid, name in ARCHETYPE_NAMES.items()}
for i, row in enumerate(rows):
    c = arch_col_map.get(row["Archetype"], "#ffffff")
    for j in range(len(df_show.columns)):
        tbl[i+1, j].set_facecolor(c + "33")  # light tint

ax.set_title("Player Lookup Database — Notable 2024-25 Prospects",
             fontsize=13, fontweight="bold", pad=10)
save(fig, "07_player_lookup_showcase.png")

# ─────────────────────────────────────────────────────────────────────────────
# 08 · WNBA actual salary distribution by archetype (box plot)
# ─────────────────────────────────────────────────────────────────────────────
print("08 — WNBA salary by archetype...")
sal_26["player_norm"] = sal_26["player_full_name"].apply(normalize_name)
wnba["player_norm"] = wnba["PLAYER_NAME"].apply(normalize_name)
wnba_sal = pd.merge(wnba, sal_26[["player_norm","base_salary_2026"]], on="player_norm", how="left")
wnba_sal = wnba_sal.dropna(subset=["base_salary_2026","archetype_id"])

fig, ax = plt.subplots(figsize=(11, 6))
order = [2, 0, 3, 4, 1]  # high to low median
data_by_arch = [wnba_sal[wnba_sal["archetype_id"]==cid]["base_salary_2026"].values for cid in order]
names_ordered = [ARCHETYPE_NAMES[cid] for cid in order]
colors_ordered = [COLORS[cid] for cid in order]

bp = ax.boxplot(data_by_arch, patch_artist=True, vert=True, widths=0.5,
                medianprops=dict(color="white", lw=2.5))
for patch, color in zip(bp["boxes"], colors_ordered):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

# Overlay individual points
for i, (data, color) in enumerate(zip(data_by_arch, colors_ordered)):
    jitter = np.random.normal(0, 0.07, size=len(data))
    ax.scatter(i + 1 + jitter, data, color=color, alpha=0.6, s=30, zorder=5, edgecolors="white", lw=0.3)

ax.set_xticks(range(1, len(order)+1))
ax.set_xticklabels(names_ordered, rotation=15, ha="right", fontsize=9.5)
ax.set_ylabel("2026 Base Salary ($)", fontsize=11)
ax.set_title("Actual 2026 WNBA Salary Distribution by Archetype", fontsize=14, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1e3:.0f}k"))
ax.grid(True, axis="y", alpha=0.2)
save(fig, "08_wnba_salary_by_archetype.png")

print("\nAll visualizations saved to data/viz/")
