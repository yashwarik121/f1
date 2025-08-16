# app.py — F1 Dashboard (PDF + CSV only, no HTML export)
import os
import tempfile
from datetime import datetime
import streamlit as st
import pandas as pd
import fastf1
import plotly.express as px
import plotly.graph_objects as go

# try imports for PDF generation (kaleido + reportlab)
_kaleido_ok = True
_reportlab_ok = True
try:
    import kaleido  # noqa: F401
except Exception:
    _kaleido_ok = False
try:
    from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
except Exception:
    _reportlab_ok = False

# -------------------------
# CONFIG
# -------------------------
CACHE_DIR = "f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

st.set_page_config(page_title="F1 Analytics — PDF + CSV", layout="wide", initial_sidebar_state="expanded")
st.title("🏁 F1 WARZ Analytics Dashboard")

# Team color dictionary (extend if needed)
TEAM_COLORS = {
    "Red Bull Racing": "#0600EF",
    "Mercedes": "#00D2BE",
    "Ferrari": "#DC0000",
    "McLaren": "#FF8700",
    "Alpine": "#0090FF",
    "Aston Martin": "#2B6E4A",
    "Williams": "#005AFF",
    "AlphaTauri": "#2B2B2B",
    "Sauber": "#900000", # Alfa Romeo changed to Sauber
    "Haas": "#B6B6B6",
}
FALLBACK_COLOR = "#7f7f7f"

RACE_LIST = [
    "Bahrain", "Saudi Arabian", "Australian", "Azerbaijan", "Miami",
    "Monaco", "Spanish", "Canadian", "Austrian", "British",
    "Hungarian", "Belgian", "Dutch", "Italian", "Singapore",
    "Japanese", "Qatar", "United States", "Mexico City",
    "São Paulo", "Las Vegas", "Abu Dhabi"
]
SESSION_MAP = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Qualifying": "Q", "Race": "R"}

# Initialize session state for delta comparison
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'd1_delta' not in st.session_state:
    st.session_state.d1_delta = None
if 'd2_delta' not in st.session_state:
    st.session_state.d2_delta = None
if 'fig_delta_plot' not in st.session_state:
    st.session_state.fig_delta_plot = None
if 'selected_drivers' not in st.session_state:
    st.session_state.selected_drivers = []
    
# -------------------------
# HELPERS
# -------------------------
def team_color(team):
    """Returns the team's primary color or a fallback color."""
    if not isinstance(team, str):
        return FALLBACK_COLOR
    for k, v in TEAM_COLORS.items():
        if k.lower() in team.lower() or team.lower() in k.lower():
            return v
    return FALLBACK_COLOR

@st.cache_resource
def safe_load_session(year, gp, s_type):
    """Safely loads an F1 session from fastf1, with caching."""
    try:
        sess = fastf1.get_session(year, gp, s_type)
        sess.load(telemetry=False, weather=True) # Telemetry takes a long time, disabling for initial load
        return sess, None
    except Exception as e:
        return None, str(e)

@st.cache_data(show_spinner="Cleaning and processing lap data...")
def laps_to_clean_df(_laps):
    """Cleans and processes lap data into a Pandas DataFrame."""
    df = pd.DataFrame(_laps)
    if df.empty:
        return pd.DataFrame()
    wanted = ['Driver', 'Team', 'LapNumber', 'LapTime', 'LapTimeSec',
              'Sector1Time', 'Sector2Time', 'Sector3Time',
              'Compound', 'Stint', 'PitInTime', 'PitOutTime', 'TyreLife', 'Position']
    available = [c for c in wanted if c in df.columns]
    if not available:
        return pd.DataFrame()
    df = df[available].copy()
    if 'LapNumber' not in df.columns:
        df['LapNumber'] = df.groupby('Driver').cumcount() + 1
    if 'LapTimeSec' not in df.columns and 'LapTime' in df.columns:
        try:
            df['LapTimeSec'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
        except Exception:
            df['LapTimeSec'] = pd.to_numeric(df['LapTime'], errors='coerce')
    for s in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
        if s in df.columns:
            try:
                df[s + 'Sec'] = pd.to_timedelta(df[s]).dt.total_seconds()
            except Exception:
                df[s + 'Sec'] = pd.to_numeric(df[s], errors='coerce')
    if 'LapTimeSec' in df.columns:
        df = df.dropna(subset=['LapTimeSec'])
        df = df[df['LapTimeSec'] > 3]
    df['Driver'] = df['Driver'].astype(str)
    df['Team'] = df['Team'].astype(str)
    return df

# FIGURE BUILDERS (with caching)
@st.cache_data
def fig_team_avg(df):
    if 'LapTimeSec' not in df.columns: return None
    team_avg = df.groupby('Team', as_index=False)['LapTimeSec'].mean().sort_values('LapTimeSec')
    color_map = {t: team_color(t) for t in team_avg['Team'].unique()}
    fig = px.bar(team_avg, x='Team', y='LapTimeSec',
                 title='Average Lap Time per Team (s)',
                 color='Team', color_discrete_map=color_map, text=team_avg['LapTimeSec'].round(2))
    fig.update_layout(showlegend=False, height=420)
    return fig

@st.cache_data
def fig_lap_progression(df):
    if 'LapTimeSec' not in df.columns: return None
    counts = df['Driver'].value_counts()
    drivers_ok = counts[counts >= 3].index.tolist()
    dfp = df[df['Driver'].isin(drivers_ok)].copy()
    if dfp.empty: return None
    fig = go.Figure()
    driver_order = sorted(dfp['Driver'].unique(), key=lambda d: dfp[dfp['Driver'] == d]['LapTimeSec'].min())
    for drv in driver_order:
        sub = dfp[dfp['Driver'] == drv].sort_values('LapNumber')
        c = team_color(sub['Team'].iloc[0]) if 'Team' in sub.columns else FALLBACK_COLOR
        fig.add_trace(go.Scatter(x=sub['LapNumber'], y=sub['LapTimeSec'], mode='lines+markers',
                                 name=drv, line=dict(color=c), marker=dict(size=4)))
    fig.update_layout(xaxis_title='Lap Number', yaxis_title='Lap Time (s)', height=450,
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig

@st.cache_data
def fig_stint_gantt(df):
    segs = []
    if 'Stint' not in df.columns:
        df['Stint'] = df.groupby('Driver').cumcount() + 1
    for driver, g in df.groupby('Driver'):
        for stint, sg in g.groupby('Stint'):
            start = int(sg['LapNumber'].min())
            end = int(sg['LapNumber'].max())
            comp = sg['Compound'].mode().iloc[0] if 'Compound' in sg.columns and not sg['Compound'].mode().empty else 'unknown'
            segs.append({'Driver': driver, 'Start': start, 'End': end, 'Compound': comp})
    if not segs: return None, {}
    segdf = pd.DataFrame(segs)
    drivers_sorted = sorted(segdf['Driver'].unique())
    fig = go.Figure()
    comp_colors = {}
    palette = px.colors.qualitative.Dark24
    pi = 0
    for comp in sorted(segdf['Compound'].unique()):
        comp_colors[comp] = palette[pi % len(palette)]; pi += 1
    for drv in drivers_sorted:
        sgs = segdf[segdf['Driver'] == drv].sort_values('Start')
        for _, r in sgs.iterrows():
            comp = r['Compound']
            fig.add_trace(go.Bar(x=[r['End'] - r['Start'] + 1], y=[drv], base=[r['Start'] - 1], orientation='h',
                                 marker=dict(color=comp_colors[comp]),
                                 hovertemplate=f"Driver: {drv}<br>Stint: Laps {r['Start']}-{r['End']}<br>Compound: {comp}<extra></extra>",
                                 name=comp, showlegend=False))
    fig.update_layout(barmode='stack', xaxis_title='Lap Number', yaxis_title='Driver', height=600,
                      legend_title="Tyre Compound")
    return fig, comp_colors

@st.cache_data
def detect_pits(df):
    pits = []
    if 'PitInTime' in df.columns or 'PitOutTime' in df.columns:
        pitrows = df[df.get('PitInTime').notna() | df.get('PitOutTime').notna()]
        for _, r in pitrows.iterrows():
            pits.append({'Driver': r['Driver'], 'PitLap': int(r['LapNumber']), 'PitIn': r.get('PitInTime'), 'PitOut': r.get('PitOutTime')})
    else:
        if 'Stint' not in df.columns:
            df['Stint'] = df.groupby('Driver').cumcount() + 1
        for driver, g in df.groupby('Driver'):
            prev_stint = None
            for _, row in g.sort_values('LapNumber').iterrows():
                if prev_stint is None:
                    prev_stint = row['Stint']
                elif row['Stint'] != prev_stint:
                    pits.append({'Driver': row['Driver'], 'PitLap': int(row['LapNumber']), 'PitIn': 'Inferred', 'PitOut': 'Inferred'})
                    prev_stint = row['Stint']
    return pd.DataFrame(pits)

@st.cache_data
def fig_pit_timeline(pits):
    if pits.empty: return None
    fig = px.scatter(pits, x='PitLap', y='Driver', color='Driver',
                     hover_data=['PitLap', 'PitIn', 'PitOut'],
                     title='Pit Laps per Driver', height=400)
    fig.update_layout(xaxis_title='Lap Number', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig

@st.cache_data
def fig_sector_breakdown(df):
    sector_cols = [c for c in df.columns if c.endswith('Sec') and 'Sector' in c]
    if not sector_cols: return None
    df_sect = df.groupby('Driver')[sector_cols].mean().reset_index()
    m = df_sect.melt(id_vars='Driver', value_vars=sector_cols, var_name='Sector', value_name='Seconds')
    m['Sector'] = m['Sector'].str.replace('Sec', '')
    fig = px.bar(m, x='Driver', y='Seconds', color='Sector', barmode='group',
                 title='Average Sector Times per Driver')
    return fig

@st.cache_data
def fig_delta(df, d1, d2):
    if d1 == d2: return None
    a = df[df['Driver'] == d1][['LapNumber', 'LapTimeSec']].rename(columns={'LapTimeSec': 'T1'})
    b = df[df['Driver'] == d2][['LapNumber', 'LapTimeSec']].rename(columns={'LapTimeSec': 'T2'})
    merged = pd.merge(a, b, on='LapNumber', how='inner')
    if merged.empty: return None
    merged['Delta'] = merged['T1'] - merged['T2']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged['LapNumber'], y=merged['Delta'], mode='lines', name=f"{d1} - {d2} (s)"))
    fig.update_layout(title=f"Lap-by-lap Delta: {d1} minus {d2}", xaxis_title='Lap', yaxis_title='Delta (s)',
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig

@st.cache_data
def fig_position_change(df):
    if 'Position' not in df.columns: return None
    pos = df[['Driver', 'LapNumber', 'Position', 'Team']].copy()
    pos = pos.dropna(subset=['Position'])
    fig = go.Figure()
    for drv in pos['Driver'].unique():
        sub = pos[pos['Driver'] == drv].sort_values('LapNumber')
        c = team_color(sub['Team'].iloc[0]) if 'Team' in sub.columns else FALLBACK_COLOR
        fig.add_trace(go.Scatter(x=sub['LapNumber'], y=sub['Position'], mode='lines+markers', name=drv,
                                 line=dict(color=c), marker=dict(size=4)))
    fig.update_layout(title='Position by Lap', xaxis_title='Lap', yaxis_title='Position',
                      yaxis_autorange='reversed', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig

@st.cache_data
def fig_fastest_lap_heatmap(df):
    if 'LapTimeSec' not in df.columns: return None
    piv = df.pivot_table(index='Driver', columns='LapNumber', values='LapTimeSec', aggfunc='min')
    if piv.empty: return None
    piv_norm = piv.apply(lambda row: row / row.min(), axis=1)
    fig = px.imshow(piv_norm, labels=dict(x='Lap', y='Driver', color='Relative Pace'),
                    title='Fastest-lap Heatmap (normalized)')
    return fig

# -------------------------
# UI
# -------------------------
with st.sidebar:
    st.header("About This App")
    with st.expander("Expand to read"):
        st.markdown(
            """
            This dashboard is an open-source tool for analyzing Formula 1 race data.
            It uses the `fastf1` library to fetch and process data, and `Streamlit`
            to create an interactive web interface.

            **Features:**
            - **Visualize Race Data:** Explore lap times, pit stops, and more.
            - **Compare Drivers:** Use the delta chart to see lap-by-lap differences.
            - **Export Reports:** Generate a PDF report of all the charts or download a CSV of the raw data.
            """
        )

    st.header("Session Selection")
    with st.form("session_form"):
        year = st.number_input("Season", min_value=2018, max_value=datetime.now().year, value=2023)
        gp = st.selectbox("Grand Prix", RACE_LIST, index=RACE_LIST.index("Monaco"))
        session_label = st.selectbox("Session", ["Practice 1", "Practice 2", "Practice 3", "Qualifying", "Race"])
        session_type = SESSION_MAP[session_label]
        
        load_button = st.form_submit_button("Load Race Data")

st.markdown("---")

if load_button:
    with st.spinner(f"Loading session data for {gp} {year} ({session_label})... (This can take a few minutes the first time.)"):
        sess, err = safe_load_session(year, gp, session_type)
    
    if err:
        st.error(f"Failed to load session: {err}")
        st.session_state.df = pd.DataFrame()
    else:
        st.success(f"Successfully loaded: **{year} {gp} ({session_label})**")
        laps = sess.laps
        st.session_state.df = laps_to_clean_df(laps)
        st.session_state.selected_drivers = sorted(st.session_state.df['Driver'].unique())
        
        # Reset delta state on new session load
        st.session_state.d1_delta = None
        st.session_state.d2_delta = None
        st.session_state.fig_delta_plot = None

if not st.session_state.df.empty:
    df = st.session_state.df
    # top metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Teams", df['Team'].nunique(), help="Number of unique teams in the session.")
    with c2:
        st.metric("Drivers", df['Driver'].nunique(), help="Number of unique drivers in the session.")
    with c3:
        st.metric("Lap Records", len(df), help="Total number of recorded laps.")
    
    st.markdown("---")

    # Team avg
    st.subheader("Team Performance Overview")
    fig1 = fig_team_avg(df)
    if fig1:
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("Team average lap time data not available.")
    
    st.markdown("---")

    # Lap progression
    st.subheader("Lap Time Progression")
    fig2 = fig_lap_progression(df)
    if fig2:
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Lap time progression data not available.")

    st.markdown("---")

    # Tyre stints / Gantt
    st.subheader("Tyre Strategies and Stints")
    gantt_res = fig_stint_gantt(df)
    if gantt_res and gantt_res[0]:
        fig_gantt, comp_colors = gantt_res
        st.plotly_chart(fig_gantt, use_container_width=True)
        if comp_colors:
            st.write("---")
            st.write("**Tyre Compounds Legend:**")
            legend_html = "<div style='display:flex; flex-wrap: wrap;'>"
            for comp, col in comp_colors.items():
                legend_html += f"<div style='margin: 6px; display: flex; align-items: center;'><div style='width: 18px; height: 12px; background:{col}; margin-right: 6px; border: 1px solid #333;'></div>{comp}</div>"
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)
    else:
        st.info("Tyre stint data not available.")

    st.markdown("---")

    # Pit stops
    st.subheader("Pit Stop Analysis")
    pits = detect_pits(df)
    if not pits.empty:
        col_pits1, col_pits2 = st.columns(2)
        with col_pits1:
            fig_pc = px.bar(pits.groupby('Driver').size().reset_index(name='PitCount').sort_values('PitCount', ascending=False),
                             x='Driver', y='PitCount', title='Pit Stops per Driver')
            st.plotly_chart(fig_pc, use_container_width=True)
        with col_pits2:
            fig_pt = fig_pit_timeline(pits)
            if fig_pt:
                st.plotly_chart(fig_pt, use_container_width=True)
        
        with st.expander("View Raw Pit Stop Data"):
            st.dataframe(pits.sort_values(['Driver', 'PitLap']).head(200), use_container_width=True)
    else:
        st.info("No pit stop records detected for this session.")

    st.markdown("---")

    # Advanced analytics in an expander
    with st.expander("Show Advanced Analytics and Driver Comparisons"):
        st.subheader("Advanced Analytics")

        # Sector Breakdown
        fig_sector = fig_sector_breakdown(df)
        if fig_sector:
            st.plotly_chart(fig_sector, use_container_width=True)
        else:
            st.info("Sector data not available. This is common for some sessions.")
        
        st.markdown("---")

        # Delta comparison
        st.markdown("#### Driver Delta Comparison")
        drivers = st.session_state.selected_drivers
        
        if len(drivers) >= 2:
            # Use a form to collect user inputs for delta comparison
            with st.form("delta_form"):
                delta_cols = st.columns(2)
                
                # Set default index for selectboxes.
                default_d1_index = drivers.index(st.session_state.d1_delta) if st.session_state.d1_delta in drivers else 0
                default_d2_index = drivers.index(st.session_state.d2_delta) if st.session_state.d2_delta in drivers else min(1, len(drivers) - 1)
                
                d1 = delta_cols[0].selectbox("Driver 1", drivers, index=default_d1_index, key="delta_d1")
                d2 = delta_cols[1].selectbox("Driver 2", drivers, index=default_d2_index, key="delta_d2")
                
                delta_button = st.form_submit_button("Generate Delta Chart")
            
            # Check if the button was clicked and update session state
            if delta_button:
                st.session_state.d1_delta = st.session_state.delta_d1
                st.session_state.d2_delta = st.session_state.delta_d2
                st.session_state.fig_delta_plot = fig_delta(df, st.session_state.d1_delta, st.session_state.d2_delta)
            
            # Display the chart from session state outside the form
            if st.session_state.fig_delta_plot:
                st.plotly_chart(st.session_state.fig_delta_plot, use_container_width=True)
            else:
                st.info("Select two drivers and click 'Generate Delta Chart' to see the comparison.")
        else:
            st.info("Not enough drivers for delta comparison.")
        
        st.markdown("---")
        
        # Position Change
        fig_pos = fig_position_change(df)
        if fig_pos:
            st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.info("Position data not available. This is common for practice sessions.")
        
        st.markdown("---")

        # Fastest Lap Heatmap
        fig_heat = fig_fastest_lap_heatmap(df)
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Heatmap not available.")

    st.markdown("---")

    # Download section
    st.subheader("Export Data and Reports")
    col_download1, col_download2 = st.columns(2)

    # CSV download
    csv = df.to_csv(index=False).encode('utf-8')
    col_download1.download_button(
        label="Download Cleaned Lap Data (CSV)",
        data=csv,
        file_name=f"{gp}_{year}_{session_type}_laps.csv",
        mime='text/csv',
        use_container_width=True
    )

    # PDF export
    if _kaleido_ok and _reportlab_ok:
        if col_download2.button("Download PDF Report (charts + meta)", use_container_width=True):
            with st.spinner("Generating PDF..."):
                try:
                    # Re-create figures with selected parameters for the PDF
                    fig_delta_pdf = fig_delta(df, st.session_state.d1_delta, st.session_state.d2_delta) if st.session_state.d1_delta and st.session_state.d2_delta else None
                    
                    images_with_captions = []
                    figs_to_save = [
                        (fig_team_avg(df), 'Average lap time per team'),
                        (fig_lap_progression(df), 'Lap time progression for drivers'),
                        (gantt_res[0], 'Tyre stints (Gantt-style)'),
                        (fig_pit_timeline(pits), 'Pit stop timeline'),
                        (fig_sector_breakdown(df), 'Average sector times per driver'),
                        (fig_delta_pdf, f'Lap-by-lap delta: {st.session_state.d1_delta} minus {st.session_state.d2_delta}'),
                        (fig_position_change(df), 'Position by lap'),
                        (fig_fastest_lap_heatmap(df), 'Normalized fastest-lap heatmap')
                    ]

                    # Filter out None figures and create images
                    for fig, caption in figs_to_save:
                        if fig:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                fig.write_image(tmp.name, engine='kaleido', format='png', scale=2)
                                images_with_captions.append((tmp.name, caption))
                    
                    pdf_path = os.path.join(tempfile.gettempdir(), f"f1_report_{gp}_{year}_{session_type}.pdf")
                    meta = f"{gp} — {year} ({session_label}) | Generated: {datetime.utcnow().isoformat()} UTC"
                    
                    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
                    styles = getSampleStyleSheet()
                    flow = []
                    flow.append(Paragraph(meta, styles['Title']))
                    flow.append(Spacer(1, 12))
                    
                    for imgp, cap in images_with_captions:
                        flow.append(Image(imgp, width=500, height=300))
                        flow.append(Paragraph(cap, styles['Normal']))
                        flow.append(Spacer(1, 12))
                    
                    doc.build(flow)

                    with open(pdf_path, 'rb') as f:
                        col_download2.download_button("Download PDF", f.read(), file_name=os.path.basename(pdf_path), use_container_width=True)
                
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
                finally:
                    for imgp, _ in images_with_captions:
                        os.remove(imgp)
    else:
        col_download2.warning(f"Cannot create PDF report: missing packages. Install with: `pip install {' '.join(['kaleido', 'reportlab'] if not _kaleido_ok or not _reportlab_ok else [])}`")
