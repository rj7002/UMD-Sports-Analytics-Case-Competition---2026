import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="2026 WNBA Draft Scout",
    page_icon="🏀",
    layout="wide",
)

# ── Color palette ──────────────────────────────────────────────────────────────
NAVY  = "#0D1B2A"
BLUE  = "#1565C0"
LBLUE = "#90CAF9"
WHITE = "#FFFFFF"
LGRAY = "#E8EDF5"   # more blue-tinted so white cards pop clearly
CARD  = "#FFFFFF"
GRID  = "#DDE4EF"

ARCHETYPE_COLORS = {
    "Elite Two-Way Big":       "#D32F2F",   # strong red
    "High-Volume Wing/Guard":  "#E65100",   # deep orange — readable on white
    "Interior Specialist":     "#00796B",   # teal
    "Efficient Role Player":   "#6A1B9A",   # deep purple
    "Bench Guard/Wing":        "#0277BD",   # steel blue — distinct from nav
}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* page background */
  .stApp {{ background-color: {LGRAY}; }}
  [data-testid="stSidebar"] {{ background-color: {WHITE}; border-right: 1px solid {GRID}; }}

  /* top header bar */
  .header-bar {{
    background: {NAVY};
    padding: 1.1rem 2rem 0.9rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.4rem;
  }}
  .header-bar h1 {{
    color: {WHITE};
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    font-family: 'Calibri', sans-serif;
  }}
  .header-bar p {{
    color: {LBLUE};
    margin: 0.2rem 0 0 0;
    font-size: 0.95rem;
  }}

  /* metric cards */
  .metric-row {{ display: flex; gap: 1rem; margin-bottom: 1.4rem; }}
  .metric-card {{
    background: {WHITE};
    border-radius: 10px;
    padding: 1rem 1.4rem;
    flex: 1;
    border-left: 5px solid {BLUE};
    box-shadow: 0 2px 8px rgba(13,27,42,0.10);
  }}
  .metric-card .label {{
    font-size: 0.75rem;
    color: #788B9E;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
  }}
  .metric-card .value {{
    font-size: 1.65rem;
    font-weight: 700;
    color: {NAVY};
    margin-top: 0.1rem;
    line-height: 1.2;
  }}
  .metric-card .sub {{
    font-size: 0.8rem;
    color: #788B9E;
    margin-top: 0.1rem;
  }}

  /* archetype badge */
  .badge {{
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    color: white;
  }}

  /* player card */
  .player-card {{
    background: {WHITE};
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    box-shadow: 0 2px 10px rgba(13,27,42,0.10);
    border-top: 5px solid {BLUE};
  }}
  .player-card h2 {{ color: {NAVY}; margin: 0 0 0.2rem 0; font-size: 1.4rem; }}
  .player-card .team {{ color: #5A6E85; font-size: 0.9rem; margin-bottom: 1rem; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.8rem; }}
  .stat-box {{
    background: {LGRAY};
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    text-align: center;
    border: 1px solid {GRID};
  }}
  .stat-box .s-label {{ font-size: 0.68rem; color: #5A6E85; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em; }}
  .stat-box .s-val {{ font-size: 1.1rem; font-weight: 700; color: {NAVY}; }}

  /* section headers */
  .section-header {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {NAVY};
    border-bottom: 2px solid {BLUE};
    padding-bottom: 0.3rem;
    margin: 1.2rem 0 0.8rem 0;
  }}

  /* hide streamlit chrome */
  #MainMenu, footer {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("final_player_lookup.csv")
    df = df.drop(columns=["Player"], errors="ignore")
    df["Draft_Prob_Pct"] = (df["Draft_Prob"] * 100).round(2)
    df["proj_salary_mid_fmt"] = df["proj_salary_mid"].apply(lambda x: f"${x:,.0f}")
    df["salary_range"] = df.apply(
        lambda r: f"${r.proj_salary_low:,.0f} – ${r.proj_salary_high:,.0f}", axis=1
    )
    df["archetype_color"] = df["archetype_name"].map(ARCHETYPE_COLORS)
    for pct_col in ["fg_pct", "fg3_pct", "ft_pct", "ts_pct", "usg_pct", "ast_pct", "reb_pct"]:
        df[pct_col] = (df[pct_col] * 100).round(1)
    df["pts_per_game"] = df["pts_per_game"].round(1)
    df["blk_per_game"] = df["blk_per_game"].round(2)
    df["stl_per_game"] = df["stl_per_game"].round(2)
    return df

df = load_data()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <h1>🏀 2026 WNBA Draft Scout</h1>
  <p>Player archetypes · Draft probabilities · Salary projections — 2025-26 NCAA Seniors</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='color:{NAVY};font-size:1.1rem;font-weight:700;margin-bottom:1rem;'>Filters</div>", unsafe_allow_html=True)

    search = st.text_input("Search player", placeholder="e.g. Lauren Betts")

    archetypes = sorted(df["archetype_name"].unique())
    selected_archetypes = st.multiselect(
        "Archetype",
        options=archetypes,
        default=archetypes,
    )

    positions = sorted(df["position"].dropna().unique())
    selected_positions = st.multiselect(
        "Position",
        options=positions,
        default=positions,
    )

    min_prob = st.slider(
        "Min Draft Probability (%)",
        min_value=0.0,
        max_value=float(df["Draft_Prob_Pct"].max()),
        value=0.0,
        step=0.1,
        format="%.1f%%",
    )

    min_pts = st.slider(
        "Min Points Per Game",
        min_value=0.0,
        max_value=float(df["pts_per_game"].max()),
        value=0.0,
        step=0.5,
    )

    sort_by = st.selectbox(
        "Sort by",
        options=["Draft_Prob_Pct", "pts_per_game", "ts_pct", "reb_pct", "proj_salary_mid"],
        format_func=lambda x: {
            "Draft_Prob_Pct": "Draft Probability",
            "pts_per_game": "Points Per Game",
            "ts_pct": "True Shooting %",
            "reb_pct": "Rebound %",
            "proj_salary_mid": "Projected Salary",
        }[x],
    )

    st.markdown("---")
    st.markdown(f"<div style='color:#788B9E;font-size:0.78rem;'>UMD Sports Analytics<br>Case Competition · April 2026</div>", unsafe_allow_html=True)

# ── Apply filters ──────────────────────────────────────────────────────────────
filtered = df.copy()
if search:
    filtered = filtered[filtered["player_name"].str.contains(search, case=False, na=False)]
filtered = filtered[filtered["archetype_name"].isin(selected_archetypes)]
filtered = filtered[filtered["position"].isin(selected_positions)]
filtered = filtered[filtered["Draft_Prob_Pct"] >= min_prob]
filtered = filtered[filtered["pts_per_game"] >= min_pts]
filtered = filtered.sort_values(sort_by, ascending=False).reset_index(drop=True)

# ── Metric cards ───────────────────────────────────────────────────────────────
top1 = filtered.iloc[0]["player_name"] if len(filtered) > 0 else "—"
avg_prob = filtered["Draft_Prob_Pct"].mean() if len(filtered) > 0 else 0
top_prob = filtered["Draft_Prob_Pct"].max() if len(filtered) > 0 else 0
likely_drafted = (filtered["Draft_Prob_Pct"] >= 10).sum()

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="label">Players Shown</div>
    <div class="value">{len(filtered)}</div>
    <div class="sub">of {len(df)} total seniors</div>
  </div>
  <div class="metric-card">
    <div class="label">Top Prospect</div>
    <div class="value" style="font-size:1.15rem;">{top1}</div>
    <div class="sub">highest draft probability</div>
  </div>
  <div class="metric-card">
    <div class="label">Top Draft Prob</div>
    <div class="value">{top_prob:.1f}%</div>
    <div class="sub">in current filter</div>
  </div>
  <div class="metric-card">
    <div class="label">Likely Draftees (≥10%)</div>
    <div class="value">{likely_drafted}</div>
    <div class="sub">avg prob: {avg_prob:.2f}%</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout: table + chart ─────────────────────────────────────────────────
col_table, col_chart = st.columns([3, 2], gap="large")

with col_table:
    st.markdown('<div class="section-header">Player Rankings</div>', unsafe_allow_html=True)

    display_cols = {
        "player_name": "Player",
        "team": "School",
        "position": "Pos",
        "archetype_name": "Archetype",
        "Draft_Prob_Pct": "Draft % ",
        "pts_per_game": "PTS",
        "ts_pct": "TS%",
        "reb_pct": "REB%",
        "proj_salary_mid_fmt": "Proj. Salary",
    }

    table_df = filtered[list(display_cols.keys())].rename(columns=display_cols).head(200)

    # Shorten school names for display
    table_df["School"] = table_df["School"].str.replace(r" Women's$", "", regex=True)

    st.dataframe(
        table_df,
        use_container_width=True,
        height=460,
        column_config={
            "Player": st.column_config.TextColumn(width="medium"),
            "School": st.column_config.TextColumn(width="large"),
            "Archetype": st.column_config.TextColumn(width="medium"),
            "Draft % ": st.column_config.ProgressColumn(
                format="%.2f%%",
                min_value=0,
                max_value=100,
                width="small",
            ),
            "PTS": st.column_config.NumberColumn(format="%.1f", width="small"),
            "TS%": st.column_config.NumberColumn(format="%.1f", width="small"),
            "REB%": st.column_config.NumberColumn(format="%.1f", width="small"),
            "Proj. Salary": st.column_config.TextColumn(width="medium"),
        },
        hide_index=True,
    )

with col_chart:
    st.markdown('<div class="section-header">Draft Probability by Archetype</div>', unsafe_allow_html=True)

    plot_df = filtered[filtered["Draft_Prob_Pct"] > 0].copy()

    if len(plot_df) > 0:
        fig = px.scatter(
            plot_df,
            x="pts_per_game",
            y="Draft_Prob_Pct",
            color="archetype_name",
            color_discrete_map=ARCHETYPE_COLORS,
            hover_name="player_name",
            hover_data={
                "team": True,
                "ts_pct": ":.1f",
                "Draft_Prob_Pct": ":.2f",
                "proj_salary_mid_fmt": True,
                "archetype_name": False,
                "pts_per_game": False,
            },
            labels={
                "pts_per_game": "Points Per Game",
                "Draft_Prob_Pct": "Draft Probability (%)",
                "archetype_name": "Archetype",
                "team": "School",
                "ts_pct": "TS%",
                "proj_salary_mid_fmt": "Proj. Salary",
            },
            size_max=12,
        )
        fig.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.5, color="white")))
        fig.update_layout(
            plot_bgcolor="#F4F7FB",
            paper_bgcolor=WHITE,
            font=dict(family="Calibri, sans-serif", color=NAVY),
            legend=dict(
                title="",
                orientation="v",
                x=1.01,
                y=1,
                font=dict(size=11),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=GRID,
                borderwidth=1,
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID),
            yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID),
            height=430,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No players match the current filters.")

# ── Archetype breakdown bar chart ──────────────────────────────────────────────
st.markdown('<div class="section-header">Archetype Distribution & Average Draft Probability</div>', unsafe_allow_html=True)

col_bar1, col_bar2 = st.columns(2, gap="large")

with col_bar1:
    arch_counts = (
        filtered.groupby("archetype_name")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=True)
    )
    arch_counts["color"] = arch_counts["archetype_name"].map(ARCHETYPE_COLORS)

    fig2 = go.Figure(go.Bar(
        x=arch_counts["count"],
        y=arch_counts["archetype_name"],
        orientation="h",
        marker_color=arch_counts["color"],
        text=arch_counts["count"],
        textposition="outside",
    ))
    fig2.update_layout(
        plot_bgcolor="#F4F7FB",
        paper_bgcolor=WHITE,
        font=dict(family="Calibri, sans-serif", color=NAVY),
        margin=dict(l=10, r=50, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        height=240,
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_bar2:
    arch_prob = (
        filtered.groupby("archetype_name")["Draft_Prob_Pct"]
        .mean()
        .reset_index()
        .sort_values("Draft_Prob_Pct", ascending=True)
    )
    arch_prob["color"] = arch_prob["archetype_name"].map(ARCHETYPE_COLORS)

    fig3 = go.Figure(go.Bar(
        x=arch_prob["Draft_Prob_Pct"],
        y=arch_prob["archetype_name"],
        orientation="h",
        marker_color=arch_prob["color"],
        text=arch_prob["Draft_Prob_Pct"].apply(lambda x: f"{x:.2f}%"),
        textposition="outside",
    ))
    fig3.update_layout(
        plot_bgcolor="#F4F7FB",
        paper_bgcolor=WHITE,
        font=dict(family="Calibri, sans-serif", color=NAVY),
        margin=dict(l=10, r=70, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        height=240,
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Player detail card ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Player Detail</div>', unsafe_allow_html=True)

all_names = filtered["player_name"].tolist()
selected_name = st.selectbox(
    "Select a player for full stats",
    options=all_names,
    index=0 if all_names else None,
    label_visibility="collapsed",
)

if selected_name:
    p = filtered[filtered["player_name"] == selected_name].iloc[0]
    arch_color = ARCHETYPE_COLORS.get(p["archetype_name"], BLUE)
    school = str(p["team"]).replace(" Women's", "")

    st.markdown(f"""
    <div class="player-card">
      <h2>{p['player_name']}</h2>
      <div class="team">{school} &nbsp;·&nbsp; {p['position']} &nbsp;·&nbsp; {int(p['games_played'])} GP</div>
      <span class="badge" style="background:{arch_color}; margin-bottom:1rem; display:inline-block;">
        {p['archetype_name']}
      </span>

      <div style="margin-top:1rem;">
        <div class="stat-grid">
          <div class="stat-box">
            <div class="s-label">Draft Prob</div>
            <div class="s-val">{p['Draft_Prob_Pct']:.2f}%</div>
          </div>
          <div class="stat-box">
            <div class="s-label">Proj. Salary</div>
            <div class="s-val" style="font-size:0.95rem;">{p['salary_range']}</div>
          </div>
          <div class="stat-box">
            <div class="s-label">PTS/G</div>
            <div class="s-val">{p['pts_per_game']}</div>
          </div>
          <div class="stat-box">
            <div class="s-label">TS%</div>
            <div class="s-val">{p['ts_pct']:.1f}%</div>
          </div>
          <div class="stat-box">
            <div class="s-label">USG%</div>
            <div class="s-val">{p['usg_pct']:.1f}%</div>
          </div>
          <div class="stat-box">
            <div class="s-label">FG%</div>
            <div class="s-val">{p['fg_pct']:.1f}%</div>
          </div>
          <div class="stat-box">
            <div class="s-label">3P%</div>
            <div class="s-val">{p['fg3_pct']:.1f}%</div>
          </div>
          <div class="stat-box">
            <div class="s-label">FT%</div>
            <div class="s-val">{p['ft_pct']:.1f}%</div>
          </div>
          <div class="stat-box">
            <div class="s-label">REB%</div>
            <div class="s-val">{p['reb_pct']:.1f}%</div>
          </div>
          <div class="stat-box">
            <div class="s-label">AST%</div>
            <div class="s-val">{p['ast_pct']:.1f}%</div>
          </div>
          <div class="stat-box">
            <div class="s-label">BLK/G</div>
            <div class="s-val">{p['blk_per_game']}</div>
          </div>
          <div class="stat-box">
            <div class="s-label">STL/G</div>
            <div class="s-val">{p['stl_per_game']}</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
